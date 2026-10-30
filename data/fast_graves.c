#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_U 512

static inline float sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

static inline float softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return log1pf(expf(x));
}

static inline uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    if (x == 0) x = 1234567;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline float rand_uniform(uint32_t* seed) {
    return (float)xorshift32(seed) / 4294967296.0f;
}

static inline void rand_normal2(float* z0, float* z1, uint32_t* seed) {
    float u1 = rand_uniform(seed);
    float u2 = rand_uniform(seed);
    if (u1 < 1e-7f) u1 = 1e-7f;
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * (float)M_PI * u2;
    *z0 = r * cosf(theta);
    *z1 = r * sinf(theta);
}

int fast_sample_line(
    const float* restrict att_w, const float* restrict att_b,
    const float* restrict lstm1_w, const float* restrict lstm1_b,
    const float* restrict lstm2_w, const float* restrict lstm2_b,
    const float* restrict lstm3_w, const float* restrict lstm3_b,
    const float* restrict gmm_w, const float* restrict gmm_b,
    const float* init_c1, const float* init_h1,
    const float* init_c2, const float* init_h2,
    const float* init_c3, const float* init_h3,
    const float* init_kappa, const float* init_w,
    const float* c_one_hot, int U,
    float bias, int max_steps,
    float* out_offsets,
    uint32_t random_seed
) {
    float c1[400], h1[400];
    float c2[400], h2[400];
    float c3[400], h3[400];
    float kappa[10];
    float w[73];

    memcpy(c1, init_c1, 400 * sizeof(float));
    memcpy(h1, init_h1, 400 * sizeof(float));
    memcpy(c2, init_c2, 400 * sizeof(float));
    memcpy(h2, init_h2, 400 * sizeof(float));
    memcpy(c3, init_c3, 400 * sizeof(float));
    memcpy(h3, init_h3, 400 * sizeof(float));
    memcpy(kappa, init_kappa, 10 * sizeof(float));
    memcpy(w, init_w, 73 * sizeof(float));

    float curr_dx = 0.0f, curr_dy = 0.0f, curr_eos = 0.0f;
    uint32_t seed = random_seed ? random_seed : 42;

    float phi[MAX_U];
    float s1_in[476];
    float att_in[476];
    float s2_in[876];
    float s3_in[876];
    float gates[1600];

    int sampled_count = 0;
    int max_u = U < MAX_U ? U : MAX_U;

    for (int step = 0; step < max_steps; step++) {
        // 1. LSTM 1 input = [w (73), curr_inp (3), h1 (400)] -> 476 floats
        memcpy(s1_in, w, 73 * sizeof(float));
        s1_in[73] = curr_dx;
        s1_in[74] = curr_dy;
        s1_in[75] = curr_eos;
        memcpy(s1_in + 76, h1, 400 * sizeof(float));

        memcpy(gates, lstm1_b, 1600 * sizeof(float));
        for (int i = 0; i < 476; i++) {
            float xi = s1_in[i];
            if (xi == 0.0f) continue;
            const float* restrict wi = &lstm1_w[i * 1600];
            #pragma GCC ivdep
            for (int j = 0; j < 1600; j++) {
                gates[j] += xi * wi[j];
            }
        }
        #pragma GCC ivdep
        for (int j = 0; j < 400; j++) {
            float ig = sigmoid(gates[j]);
            float jg = tanhf(gates[j + 400]);
            float fg = sigmoid(gates[j + 800] + 1.0f);
            float og = sigmoid(gates[j + 1200]);
            float c_val = fg * c1[j] + ig * jg;
            c1[j] = c_val;
            h1[j] = og * tanhf(c_val);
        }

        // 2. Attention Window = [w (73), curr_inp (3), h1 (400)] -> 476 floats
        memcpy(att_in, w, 73 * sizeof(float));
        att_in[73] = curr_dx;
        att_in[74] = curr_dy;
        att_in[75] = curr_eos;
        memcpy(att_in + 76, h1, 400 * sizeof(float));

        float att_p[30];
        memcpy(att_p, att_b, 30 * sizeof(float));
        for (int i = 0; i < 476; i++) {
            float xi = att_in[i];
            if (xi == 0.0f) continue;
            const float* restrict wi = &att_w[i * 30];
            for (int j = 0; j < 30; j++) {
                att_p[j] += xi * wi[j];
            }
        }
        for (int j = 0; j < 30; j++) {
            att_p[j] = softplus(att_p[j]);
        }

        float alpha[10], beta_arr[10];
        for (int k = 0; k < 10; k++) {
            alpha[k] = att_p[k];
            beta_arr[k] = att_p[10 + k] < 0.01f ? 0.01f : att_p[10 + k];
            kappa[k] += att_p[20 + k] / 25.0f;
        }

        // Compute phi over U characters
        int argmax_u = 0;
        float max_phi = -1.0f;

        for (int u = 0; u < max_u; u++) {
            float sum_val = 0.0f;
            float u_f = (float)u;
            for (int k = 0; k < 10; k++) {
                float diff = kappa[k] - u_f;
                sum_val += alpha[k] * expf(-(diff * diff) / beta_arr[k]);
            }
            phi[u] = sum_val;
            if (sum_val > max_phi) {
                max_phi = sum_val;
                argmax_u = u;
            }
        }

        // Compute w = sum_u (phi[u] * c_one_hot[u]) -> 73 floats
        memset(w, 0, 73 * sizeof(float));
        for (int u = 0; u < max_u; u++) {
            float p = phi[u];
            if (p < 1e-6f) continue;
            const float* cu = &c_one_hot[u * 73];
            for (int ch = 0; ch < 73; ch++) {
                w[ch] += p * cu[ch];
            }
        }

        // 3. LSTM 2 input = [curr_inp (3), h1 (400), w (73), h2 (400)] -> 876 floats
        s2_in[0] = curr_dx;
        s2_in[1] = curr_dy;
        s2_in[2] = curr_eos;
        memcpy(s2_in + 3, h1, 400 * sizeof(float));
        memcpy(s2_in + 403, w, 73 * sizeof(float));
        memcpy(s2_in + 476, h2, 400 * sizeof(float));

        memcpy(gates, lstm2_b, 1600 * sizeof(float));
        for (int i = 0; i < 876; i++) {
            float xi = s2_in[i];
            if (xi == 0.0f) continue;
            const float* restrict wi = &lstm2_w[i * 1600];
            #pragma GCC ivdep
            for (int j = 0; j < 1600; j++) {
                gates[j] += xi * wi[j];
            }
        }
        #pragma GCC ivdep
        for (int j = 0; j < 400; j++) {
            float ig = sigmoid(gates[j]);
            float jg = tanhf(gates[j + 400]);
            float fg = sigmoid(gates[j + 800] + 1.0f);
            float og = sigmoid(gates[j + 1200]);
            float c_val = fg * c2[j] + ig * jg;
            c2[j] = c_val;
            h2[j] = og * tanhf(c_val);
        }

        // 4. LSTM 3 input = [curr_inp (3), h2 (400), w (73), h3 (400)] -> 876 floats
        s3_in[0] = curr_dx;
        s3_in[1] = curr_dy;
        s3_in[2] = curr_eos;
        memcpy(s3_in + 3, h2, 400 * sizeof(float));
        memcpy(s3_in + 403, w, 73 * sizeof(float));
        memcpy(s3_in + 476, h3, 400 * sizeof(float));

        memcpy(gates, lstm3_b, 1600 * sizeof(float));
        for (int i = 0; i < 876; i++) {
            float xi = s3_in[i];
            if (xi == 0.0f) continue;
            const float* restrict wi = &lstm3_w[i * 1600];
            #pragma GCC ivdep
            for (int j = 0; j < 1600; j++) {
                gates[j] += xi * wi[j];
            }
        }
        #pragma GCC ivdep
        for (int j = 0; j < 400; j++) {
            float ig = sigmoid(gates[j]);
            float jg = tanhf(gates[j + 400]);
            float fg = sigmoid(gates[j + 800] + 1.0f);
            float og = sigmoid(gates[j + 1200]);
            float c_val = fg * c3[j] + ig * jg;
            c3[j] = c_val;
            h3[j] = og * tanhf(c_val);
        }

        // 5. GMM Layer
        float gmm_p[121];
        memcpy(gmm_p, gmm_b, 121 * sizeof(float));
        for (int i = 0; i < 400; i++) {
            float xi = h3[i];
            if (xi == 0.0f) continue;
            const float* restrict wi = &gmm_w[i * 121];
            for (int j = 0; j < 121; j++) {
                gmm_p[j] += xi * wi[j];
            }
        }

        // Softmax mixture weights pis
        float pis[20];
        float max_pi = -1e9f;
        for (int k = 0; k < 20; k++) {
            pis[k] = gmm_p[k] * (1.0f + bias);
            if (pis[k] > max_pi) max_pi = pis[k];
        }
        float sum_pi = 0.0f;
        for (int k = 0; k < 20; k++) {
            pis[k] = expf(pis[k] - max_pi);
            sum_pi += pis[k];
        }
        float sum_clean_pi = 0.0f;
        for (int k = 0; k < 20; k++) {
            pis[k] /= sum_pi;
            if (pis[k] < 0.01f) pis[k] = 0.0f;
            sum_clean_pi += pis[k];
        }
        for (int k = 0; k < 20; k++) {
            pis[k] /= (sum_clean_pi > 1e-6f ? sum_clean_pi : 1.0f);
        }

        // Sample component k
        float r_comp = rand_uniform(&seed);
        int chosen_k = 0;
        float cum_pi = 0.0f;
        for (int k = 0; k < 20; k++) {
            cum_pi += pis[k];
            if (r_comp <= cum_pi) {
                chosen_k = k;
                break;
            }
        }

        // Sample dx, dy from 2D Gaussian
        float s1_val = expf(gmm_p[20 + chosen_k] - bias);
        if (s1_val < 1e-4f) s1_val = 1e-4f;
        if (s1_val > 1e5f) s1_val = 1e5f;

        float s2_val = expf(gmm_p[40 + chosen_k] - bias);
        if (s2_val < 1e-4f) s2_val = 1e-4f;
        if (s2_val > 1e5f) s2_val = 1e5f;

        float rho = tanhf(gmm_p[60 + chosen_k]);
        if (rho < -0.99999f) rho = -0.99999f;
        if (rho > 0.99999f) rho = 0.99999f;

        float mu1 = gmm_p[80 + chosen_k];
        float mu2 = gmm_p[100 + chosen_k];

        float z0, z1;
        rand_normal2(&z0, &z1, &seed);

        float dx = mu1 + s1_val * z0;
        float dy = mu2 + s2_val * (rho * z0 + sqrtf(1.0f - rho * rho) * z1);

        // EOS probability
        float e_prob = sigmoid(gmm_p[120]);
        if (e_prob < 0.01f) e_prob = 0.0f;
        float eos = rand_uniform(&seed) < e_prob ? 1.0f : 0.0f;

        curr_dx = dx;
        curr_dy = dy;
        curr_eos = eos;

        out_offsets[sampled_count * 3 + 0] = dx;
        out_offsets[sampled_count * 3 + 1] = dy;
        out_offsets[sampled_count * 3 + 2] = eos;
        sampled_count++;

        if (argmax_u >= U - 1 && eos == 1.0f) {
            break;
        }
    }

    return sampled_count;
}
