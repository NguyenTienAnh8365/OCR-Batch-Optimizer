"""
OCR Batch Optimizer — FastAPI Backend
Chạy: uvicorn api:app --reload --port 8000

Ported 1-to-1 từ OCRBatchOptimizer Python class.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.optimize import curve_fit
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OCR Batch Optimizer API",
    description="Worker Pool + Monte Carlo · Power-Law Fit · Exact port of OCRBatchOptimizer",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class RawPayload(BaseModel):
    """
    raw: dict[str(m), list[list[float]]]
        Mỗi key là batch-size m.
        Mỗi value là list của các runs, mỗi run là list thời gian (giây).
    """
    raw: dict[str, list[list[float]]] = Field(
        ...,
        example={
            "1": [[107], [108], [108], [108]],
            "2": [[163, 185], [142, 152], [150, 177], [147, 159]],
            "4": [[248, 280, 313, 340], [241, 282, 310, 320],
                  [229, 277, 301, 339], [232, 274, 303, 323]],
        },
    )
    warmup_runs:      int = Field(1,    ge=0, description="Số warmup runs bị bỏ")
    m_max_extrap:     int = Field(2,    ge=1, description="Nhân tối đa ngoài vùng đo")
    m_max_concurrent: int = Field(12,   ge=1, description="Giới hạn GPU concurrent (VRAM)")
    n_sim:            int = Field(1000, ge=50, le=10000, description="Số lần Monte Carlo")


class OptimizePayload(RawPayload):
    N:       int  = Field(..., ge=1, description="Số requests đang pending")
    mode:    str  = Field("pool",  pattern="^(batch|pool)$")
    use_p95: bool = Field(False,   description="True = dùng p95 thay T_max avg")


class BulkPayload(RawPayload):
    N_list:  list[int] = Field(..., description="Danh sách N cần evaluate")
    mode:    str       = Field("pool", pattern="^(batch|pool)$")
    use_p95: bool      = Field(False)


# ── Core — ported 1-to-1 ─────────────────────────────────────────────────────
class OCRBatchOptimizer:

    def __init__(self, raw: dict, warmup: int, m_max_extrap: int,
                 m_vram: int, n_sim: int):
        self.raw            = {int(k): v for k, v in raw.items()}
        self.warmup         = warmup
        self.n_sim          = n_sim
        self.m_max_measured = max(self.raw.keys())
        self.m_hard_cap     = min(self.m_max_measured * m_max_extrap, m_vram)
        self.profile        = self._summarize()
        self.model_avg      = self._fit("avg")
        self.model_max      = self._fit("max")

    # ── 1. Benchmark ─────────────────────────────────────────────────────────
    def _summarize(self) -> dict:
        profile = {}
        for m, runs in self.raw.items():
            stable = runs[self.warmup:]
            if not stable:
                raise ValueError(f"m={m}: không còn run nào sau warmup={self.warmup}")
            all_times     = [t for run in stable for t in run]
            T_avg         = float(np.mean(all_times))
            T_max_per_run = [max(run) for run in stable]
            T_max         = float(np.mean(T_max_per_run))
            # std_run: std của T_max per run (run-to-run stability) → dùng cho T_max_p95
            std_run       = float(np.std(T_max_per_run, ddof=1)) if len(stable) > 1 else 0.0
            # std_item: std của từng item riêng lẻ → dùng cho pool simulation sampling
            # Quan trọng: nếu chỉ có 1 run (vd m=10), std_run=0 nhưng std_item vẫn có giá trị
            std_item      = float(np.std(all_times, ddof=1)) if len(all_times) > 1 else 0.0
            cv            = std_run / T_max if T_max > 0 else 0.0
            profile[m]    = {
                "T_avg":    T_avg,
                "T_max":    T_max,
                "std":      std_run,   # giữ key "std" để không break export_profile / T_max_p95
                "std_item": std_item,  # thêm mới — dùng cho pool sampling
                "cv":       cv,
                "n":        len(stable),
                "warning":  bool(cv >= 0.05),
            }
        return profile

    # ── 2. Power-law fit ─────────────────────────────────────────────────────
    def _power_law(self, m, t0, alpha):
        return t0 * (m ** alpha)

    def _fit(self, kind: str) -> dict:
        ms = np.array(list(self.profile.keys()), dtype=float)
        ts = np.array([v[f"T_{kind}"] for v in self.profile.values()])
        ws = np.array([math.sqrt(v["n"]) for v in self.profile.values()])
        try:
            popt, pcov = curve_fit(
                self._power_law, ms, ts,
                p0=[100, 0.75], bounds=(0, [1e5, 2.0]),
                sigma=1 / ws, absolute_sigma=False,
            )
        except RuntimeError as e:
            raise ValueError(f"curve_fit thất bại ({kind}): {e}")
        t0, alpha = popt
        alpha_std = float(math.sqrt(max(float(pcov[1, 1]), 0.0)))
        pred      = self._power_law(ms, *popt)
        ss_res    = float(np.sum((ts - pred) ** 2))
        ss_tot    = float(np.sum((ts - ts.mean()) ** 2))
        r2        = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return {
            "t0":        float(t0),
            "alpha":     float(alpha),
            "alpha_std": alpha_std,
            "r2":        float(r2),
            "warning":   bool(r2 < 0.92 or alpha > 1.3 or alpha < 0.4),
        }

    # ── Estimators ────────────────────────────────────────────────────────────
    def T_avg(self, m: int) -> float:
        if m in self.profile:
            return self.profile[m]["T_avg"]
        # Khi T_avg fit kém (R²<0.92) nhưng T_max fit tốt,
        # dùng T_max model × ratio T_avg/T_max trung bình thay vì fit T_avg trực tiếp
        if self.model_avg["r2"] < 0.92 and self.model_max["r2"] >= 0.92:
            t_max_pred = self.model_max["t0"] * (m ** self.model_max["alpha"])
            ratios = [v["T_avg"] / v["T_max"] for v in self.profile.values() if v["T_max"] > 0]
            ratio  = float(np.mean(ratios))
            return t_max_pred * ratio
        return self.model_avg["t0"] * (m ** self.model_avg["alpha"])

    def T_max(self, m: int) -> float:
        if m in self.profile:
            return self.profile[m]["T_max"]
        return self.model_max["t0"] * (m ** self.model_max["alpha"])

    def T_max_p95(self, m: int) -> float:
        unc = (self.model_max["alpha_std"] / self.model_max["alpha"]
               if self.model_max["alpha"] > 0 else 0.2)
        if m in self.profile:
            std_run = self.profile[m]["std"]
            if std_run > 0:
                # Trường hợp bình thường: đủ runs để tính std thực
                return self.profile[m]["T_max"] + 1.645 * std_run
            else:
                # Chỉ 1 run → std_run=0, fallback sang model uncertainty
                # tránh việc m đo 1 lần trông an toàn hơn m nội suy từ nhiều điểm
                return self.profile[m]["T_max"] * (1 + 1.645 * unc)
        t_pred = self.model_max["t0"] * (m ** self.model_max["alpha"])
        return t_pred * (1 + 1.645 * unc)

    # ── Penalty ───────────────────────────────────────────────────────────────
    def penalty(self, m: int) -> float:
        if m <= self.m_max_measured:
            return 0.0
        dist       = math.log(m / self.m_max_measured) / math.log(2)
        unc        = (self.model_max["alpha_std"] / self.model_max["alpha"]
                      if self.model_max["alpha"] > 0 else 0.2)
        mem_ratio  = m / self.m_hard_cap
        saturation = 0.18 * (mem_ratio - 0.7) ** 2 if mem_ratio > 0.7 else 0.0
        return 0.025 + 0.035 * dist + 0.18 * unc + saturation

    def _penalty_breakdown(self, m: int) -> dict:
        if m <= self.m_max_measured:
            return {"base": 0.0, "distance": 0.0, "uncertainty": 0.0,
                    "vram_sat": 0.0, "total": 0.0}
        dist  = math.log(m / self.m_max_measured) / math.log(2)
        unc   = (self.model_max["alpha_std"] / self.model_max["alpha"]
                 if self.model_max["alpha"] > 0 else 0.2)
        mr    = m / self.m_hard_cap
        sat   = 0.18 * (mr - 0.7) ** 2 if mr > 0.7 else 0.0
        b, d, u = 0.025, 0.035 * dist, 0.18 * unc
        total = b + d + u + sat
        return {
            "base":        round(b   * 100, 3),
            "distance":    round(d   * 100, 3),
            "uncertainty": round(u   * 100, 3),
            "vram_sat":    round(sat * 100, 3),
            "total":       round(total * 100, 3),
        }

    # ── Batch model ───────────────────────────────────────────────────────────
    def calculate_metrics_batch(self, N: int, m: int, use_p95: bool) -> tuple:
        k        = math.ceil(N / m)
        r        = N % m
        t_avg    = self.T_avg(m)
        t_max    = self.T_max_p95(m) if use_p95 else self.T_max(m)
        makespan = k * t_max
        tput     = N / makespan
        if r == 0:
            avg_lat = t_avg + t_max * (k - 1) / 2
        else:
            lat_full = t_avg + t_max * (k - 2) / 2 if k > 1 else t_avg
            lat_last = t_avg + t_max * (k - 1)
            avg_lat  = ((k - 1) * m * lat_full + r * lat_last) / N
        return avg_lat, makespan, tput

    # ── Worker pool model (Monte Carlo) ──────────────────────────────────────
    def _sample_job_times(self, m: int, n_jobs: int) -> np.ndarray:
        mu = self.T_avg(m)
        # Dùng std_item (std của từng item riêng lẻ) thay vì std (std của T_max per run).
        # std phản ánh run-to-run stability → dùng cho T_max_p95.
        # std_item phản ánh variance thực của job times → đúng cho pool sampling.
        # Trường hợp đặc biệt: m chỉ có 1 run → std=0 nhưng std_item vẫn có giá trị.
        # Fallback 15% CV khi extrapolate (thận trọng hơn 10% cũ).
        std = self.profile[m]["std_item"] if m in self.profile else mu * 0.15
        if std <= 0 or mu <= 0:
            return np.full(n_jobs, mu)
        sigma2 = math.log(1 + (std / mu) ** 2)
        mu_ln  = math.log(mu) - sigma2 / 2
        return np.random.lognormal(mu_ln, math.sqrt(sigma2), n_jobs)

    def calculate_metrics_pool(self, N: int, m: int,
                                use_p95: bool, n_sim: int) -> tuple:
        # Khi N <= m: tất cả jobs chạy song song ngay từ đầu — không có scheduling.
        # Kết quả giống hệt 1 batch đơn, Monte Carlo không có ý nghĩa ở đây
        # và sẽ cho kết quả sai vì resample từ distribution thay vì dùng giá trị thực đo.
        if N <= m:
            makespan = self.T_max_p95(m) if use_p95 else self.T_max(m)
            avg_lat  = self.T_avg(m)
            return avg_lat, makespan, N / makespan

        avg_lats, makespans = [], []
        for _ in range(n_sim):
            job_times    = self._sample_job_times(m, N)
            worker_free  = np.zeros(m)
            finish_times = []
            for t in job_times:
                w              = int(np.argmin(worker_free))
                worker_free[w] += t
                finish_times.append(float(worker_free[w]))
            avg_lats.append(float(np.mean(finish_times)))
            makespans.append(float(np.max(finish_times)))
        if use_p95:
            avg_lat  = float(np.percentile(avg_lats,  95))
            makespan = float(np.percentile(makespans, 95))
        else:
            avg_lat  = float(np.mean(avg_lats))
            makespan = float(np.mean(makespans))
        return avg_lat, makespan, N / makespan

    # ── Find optimal ─────────────────────────────────────────────────────────
    def find_optimal(self, N: int, mode: str, use_p95: bool) -> dict:
        rows = []
        for m in range(1, min(N, self.m_hard_cap) + 1):
            if mode == "pool":
                lat, ms, tput = self.calculate_metrics_pool(N, m, use_p95, self.n_sim)
            else:
                lat, ms, tput = self.calculate_metrics_batch(N, m, use_p95)

            pen        = self.penalty(m)
            eff_lat    = lat * (1 + pen)
            p95_val    = self.T_max_p95(m) if m in self.profile else None

            rows.append({
                "m":                 m,
                "T_avg":             round(self.T_avg(m), 2),
                "T_max":             round(self.T_max(m), 2),
                "T_max_p95":         round(p95_val, 2) if p95_val is not None else None,
                "avg_lat":           round(lat, 2),
                "eff_lat":           round(eff_lat, 2),
                "makespan":          round(ms, 2),
                "throughput":        round(tput, 6),
                "penalty":           round(pen, 6),
                "penalty_pct":       round(pen * 100, 2),
                "penalty_breakdown": self._penalty_breakdown(m),
                "src":               "measured" if m in self.profile else "extrapolated",
                "is_optimal":        False,
            })

        if not rows:
            raise ValueError(f"Không tìm được m hợp lệ (N={N}, cap={self.m_hard_cap})")

        best               = min(rows, key=lambda r: r["eff_lat"])
        best["is_optimal"] = True
        return {
            "N":          N,
            "mode":       mode,
            "use_p95":    use_p95,
            "m_hard_cap": self.m_hard_cap,
            "optimal_m":  best["m"],
            "rows":       rows,
        }

    # ── Export helpers ────────────────────────────────────────────────────────
    def export_profile(self) -> dict:
        return {
            m: {**v, "T_max_p95": round(self.T_max_p95(m), 2)}
            for m, v in self.profile.items()
        }

    def fit_curve_points(self) -> list[dict]:
        return [
            {
                "m":     m,
                "T_avg": round(self.T_avg(m), 2),
                "T_max": round(self.T_max(m), 2),
            }
            for m in range(1, self.m_hard_cap + 1)
        ]


# ── Shared builder ────────────────────────────────────────────────────────────
def build(req: RawPayload) -> OCRBatchOptimizer:
    try:
        return OCRBatchOptimizer(
            raw=req.raw,
            warmup=req.warmup_runs,
            m_max_extrap=req.m_max_extrap,
            m_vram=req.m_max_concurrent,
            n_sim=req.n_sim,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["util"])
def health() -> dict[str, Any]:
    """Health check."""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/benchmark", tags=["analysis"])
def benchmark(req: RawPayload) -> dict[str, Any]:
    """
    Phân tích benchmark data thô.
    Trả về: profile mỗi m, tham số power-law fit, curve points để vẽ đồ thị.
    """
    opt = build(req)
    return {
        "profile":        opt.export_profile(),
        "model_avg":      opt.model_avg,
        "model_max":      opt.model_max,
        "m_max_measured": opt.m_max_measured,
        "m_hard_cap":     opt.m_hard_cap,
        "fit_curve":      opt.fit_curve_points(),
    }


@app.post("/optimize", tags=["optimize"])
def optimize(req: OptimizePayload) -> dict[str, Any]:
    """
    Tìm m* tối ưu cho N requests đang pending.
    Trả về bảng đầy đủ mọi m ứng viên + highlight m*.
    """
    opt = build(req)
    return opt.find_optimal(req.N, req.mode, req.use_p95)


@app.post("/optimize/bulk", tags=["optimize"])
def optimize_bulk(req: BulkPayload) -> dict[str, Any]:
    """
    Tìm m* cho nhiều giá trị N cùng lúc.
    Hữu ích để vẽ đồ thị m* = f(N).
    """
    opt     = build(req)
    results = []
    for N in req.N_list:
        try:
            r    = opt.find_optimal(N, req.mode, req.use_p95)
            best = next(row for row in r["rows"] if row["is_optimal"])
            results.append({
                "N":          N,
                "optimal_m":  r["optimal_m"],
                "avg_lat":    best["avg_lat"],
                "eff_lat":    best["eff_lat"],
                "makespan":   best["makespan"],
                "throughput": best["throughput"],
                "penalty_pct": best["penalty_pct"],
            })
        except ValueError as e:
            results.append({"N": N, "error": str(e)})
    return {"results": results, "mode": req.mode, "use_p95": req.use_p95}


@app.post("/compare", tags=["optimize"])
def compare(req: OptimizePayload) -> dict[str, Any]:
    """
    So sánh trực tiếp batch vs pool cho cùng N.
    """
    opt = build(req)
    return {
        "N":     req.N,
        "batch": opt.find_optimal(req.N, "batch", req.use_p95),
        "pool":  opt.find_optimal(req.N, "pool",  req.use_p95),
    }