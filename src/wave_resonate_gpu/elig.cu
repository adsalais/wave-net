// One thread per synapse: advance the 2-state HYPR eligibility and accumulate elig. Mirrors cpu_accrue.
// Manual abs (no <cmath>) so NVRTC compiles it standalone.
extern "C" __global__ void accrue(
    const unsigned int* tgt_g, const unsigned int* src_g,
    const float* b_eff_g, const float* omega_g, const float* psi_g, const unsigned int* prev_fired_g,
    float* eps_x, float* eps_y, float* elig,
    unsigned int n, float dt, float cut)
{
    unsigned int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    unsigned int j = tgt_g[e];
    unsigned int i = src_g[e];
    float inj = prev_fired_g[i] != 0u ? dt : 0.0f;
    float b = b_eff_g[j], om = omega_g[j], psi = psi_g[j];
    float ex = eps_x[e], ey = eps_y[e];
    float coef = 1.0f + dt * b;
    float nex = coef * ex - dt * om * ey + inj;
    float ney = dt * om * ex + coef * ey;
    if ((nex < 0.0f ? -nex : nex) < cut) nex = 0.0f;
    if ((ney < 0.0f ? -ney : ney) < cut) ney = 0.0f;
    eps_x[e] = nex; eps_y[e] = ney;
    if (psi != 0.0f && nex != 0.0f) elig[e] += psi * nex;
}
