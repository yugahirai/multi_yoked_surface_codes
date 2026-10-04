/* Box-plus (XOR) combination of log-likelihood ratios along the last axis. */
#include <math.h>
#include <stddef.h>

void boxplus_rows(const double *x, ptrdiff_t rows, ptrdiff_t k, double *out) {
    for (ptrdiff_t r = 0; r < rows; ++r) {
        const double *p = x + r * k;
        double o = p[0];
        for (ptrdiff_t j = 1; j < k; ++j) {
            double b = p[j];
            double so = (double)((o > 0.0) - (o < 0.0));
            double sb = (double)((b > 0.0) - (b < 0.0));
            double ao = fabs(o), ab = fabs(b);
            double m = ao < ab ? ao : ab;
            o = so * sb * m + log1p(exp(-fabs(o + b))) - log1p(exp(-fabs(o - b)));
        }
        out[r] = o;
    }
}
