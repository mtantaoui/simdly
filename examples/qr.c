#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Matrix access macro for column-major order
#define MAT(A, i, j, ld) A[(j) * (ld) + (i)]

// Function to compute Householder vector and scalar - CORRECTED
void housev(double *alpha11, double *a21, int m21, double *tau1)
{
    int i;
    double norm_x, sigma, beta;

    // Compute norm of a21
    sigma = 0.0;
    for (i = 0; i < m21; i++)
    {
        sigma += a21[i] * a21[i];
    }

    // Compute norm of entire vector [alpha11; a21]
    norm_x = sqrt((*alpha11) * (*alpha11) + sigma);

    if (norm_x == 0.0)
    {
        *tau1 = 0.0;
        return;
    }

    // Choose sign to avoid cancellation
    double sign_alpha = (*alpha11 >= 0.0) ? 1.0 : -1.0;
    double alpha_new = -sign_alpha * norm_x;

    // Compute beta = alpha11 - alpha_new
    beta = *alpha11 - alpha_new;

    if (fabs(beta) < 1e-15)
    {
        *tau1 = 0.0;
        return;
    }

    // Compute tau = 2 * beta^2 / (sigma + beta^2)
    *tau1 = 2.0 * beta * beta / (sigma + beta * beta);

    // Update alpha11 (this becomes the R element)
    *alpha11 = alpha_new;

    // Update a21 (these become the Householder vector elements)
    for (i = 0; i < m21; i++)
    {
        a21[i] = a21[i] / beta;
    }
}

// HQR function - CORRECTED
void HQR(double *A, int m, int n, double *t, double *A_out)
{
    int i, j, k;

    // Copy A to A_out
    memcpy(A_out, A, m * n * sizeof(double));

    // Initialize t to zeros
    for (i = 0; i < n; i++)
    {
        t[i] = 0.0;
    }

    // Main loop - process each column
    for (k = 0; k < n && k < m - 1; k++)
    {
        int m_below = m - k - 1; // Number of elements below diagonal

        if (m_below <= 0)
            continue;

        // Get pointers to current column elements
        double *alpha11 = &MAT(A_out, k, k, m);
        double *a21 = &MAT(A_out, k + 1, k, m);

        // Compute Householder vector and scalar
        housev(alpha11, a21, m_below, &t[k]);

        if (fabs(t[k]) < 1e-15)
            continue;

        // Apply Householder transformation to remaining columns
        for (j = k + 1; j < n; j++)
        {
            // Compute w = (alpha12 + a21^T * A22) * tau
            double w = MAT(A_out, k, j, m);
            for (i = 0; i < m_below; i++)
            {
                w += a21[i] * MAT(A_out, k + 1 + i, j, m);
            }
            w *= t[k];

            // Update column j: alpha12 = alpha12 - w, A22 = A22 - a21 * w
            MAT(A_out, k, j, m) -= w;
            for (i = 0; i < m_below; i++)
            {
                MAT(A_out, k + 1 + i, j, m) -= a21[i] * w;
            }
        }
    }
}

// Function to construct Q matrix from Householder vectors - CORRECTED
void construct_Q(double *A_out, double *t, int m, int n, double *Q)
{
    int i, j, k, l;

    // Initialize Q as identity matrix
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < m; j++)
        {
            MAT(Q, i, j, m) = (i == j) ? 1.0 : 0.0;
        }
    }

    // Apply Householder transformations in reverse order
    for (k = n - 1; k >= 0; k--)
    {
        if (fabs(t[k]) < 1e-15)
            continue;

        int m_below = m - k - 1;
        if (m_below <= 0)
            continue;

        // Get Householder vector v = [1; a21]
        double *v = (double *)malloc((m_below + 1) * sizeof(double));
        v[0] = 1.0;
        for (i = 0; i < m_below; i++)
        {
            v[i + 1] = MAT(A_out, k + 1 + i, k, m);
        }

        // Apply H = I - tau * v * v^T to Q
        // For each column of Q
        for (j = 0; j < m; j++)
        {
            // Compute w = v^T * Q(:,j)
            double w = 0.0;
            for (i = 0; i <= m_below; i++)
            {
                w += v[i] * MAT(Q, k + i, j, m);
            }
            w *= t[k];

            // Update Q(:,j) = Q(:,j) - w * v
            for (i = 0; i <= m_below; i++)
            {
                MAT(Q, k + i, j, m) -= w * v[i];
            }
        }

        free(v);
    }
}

// Function to extract R matrix
void extract_R(double *A_out, int m, int n, double *R)
{
    int i, j;

    // Initialize R as zeros
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            MAT(R, i, j, m) = 0.0;
        }
    }

    // Copy upper triangular part
    for (i = 0; i < m && i < n; i++)
    {
        for (j = i; j < n; j++)
        {
            MAT(R, i, j, m) = MAT(A_out, i, j, m);
        }
    }
}

// Helper function to print matrix
void print_matrix(const char *name, double *A, int m, int n)
{
    printf("%s:\n", name);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%8.4f ", MAT(A, i, j, m));
        }
        printf("\n");
    }
    printf("\n");
}

// Function to verify QR decomposition
void verify_qr(double *A, double *Q, double *R, int m, int n)
{
    double *QR = (double *)malloc(m * n * sizeof(double));
    double max_error = 0.0;

    // Compute QR
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            MAT(QR, i, j, m) = 0.0;
            for (int k = 0; k < m; k++)
            {
                MAT(QR, i, j, m) += MAT(Q, i, k, m) * MAT(R, k, j, m);
            }
        }
    }

    printf("Q * R:\n");
    print_matrix("Q * R", QR, m, n);

    // Compute error
    printf("Error (A - Q*R):\n");
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double error = MAT(A, i, j, m) - MAT(QR, i, j, m);
            printf("%8.4e ", error);
            if (fabs(error) > max_error)
                max_error = fabs(error);
        }
        printf("\n");
    }
    printf("Maximum error: %8.4e\n\n", max_error);

    // Check orthogonality of Q
    double *QTQ = (double *)malloc(m * m * sizeof(double));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            MAT(QTQ, i, j, m) = 0.0;
            for (int k = 0; k < m; k++)
            {
                MAT(QTQ, i, j, m) += MAT(Q, k, i, m) * MAT(Q, k, j, m);
            }
        }
    }

    printf("Q^T * Q (should be identity):\n");
    print_matrix("Q^T * Q", QTQ, m, m);

    free(QR);
    free(QTQ);
}

// Example usage
int main()
{
    // Example 4x3 matrix in column-major order
    int m = 4, n = 3;
    double A_colmajor[] = {
        1.0, 4.0, 7.0, 10.0, // First column
        2.0, 5.0, 8.0, 11.0, // Second column
        3.0, 6.0, 9.0, 12.0  // Third column
    };

    double *A_out = (double *)malloc(m * n * sizeof(double));
    double *t = (double *)malloc(n * sizeof(double));
    double *Q = (double *)malloc(m * m * sizeof(double));
    double *R = (double *)malloc(m * n * sizeof(double));

    printf("Original matrix A (column-major storage):\n");
    print_matrix("A", A_colmajor, m, n);

    // Perform HQR factorization
    HQR(A_colmajor, m, n, t, A_out);

    // Extract Q and R matrices
    construct_Q(A_out, t, m, n, Q);
    extract_R(A_out, m, n, R);

    printf("Q matrix:\n");
    print_matrix("Q", Q, m, m);

    printf("R matrix:\n");
    print_matrix("R", R, m, n);

    printf("Householder scalars t:\n");
    for (int i = 0; i < n; i++)
    {
        printf("t[%d] = %8.4f\n", i, t[i]);
    }
    printf("\n");

    // Verify QR = A
    printf("=== Verification ===\n");
    verify_qr(A_colmajor, Q, R, m, n);

    free(A_out);
    free(t);
    free(Q);
    free(R);

    return 0;
}