#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

//solutore LU
Vector2d solutionLU(Matrix2d A, Vector2d b)
{
	Vector2d x = Vector2d::Zero();
	
	if (A.determinant() != 0) {
		FullPivLU<Matrix2d> lu(A);
		x = lu.solve(b);
	}
	else {
		cerr << "La matrice A non è invertibile" << endl;
	}
	
	return x;
}


//solutore QR
Vector2d solutionQR(Matrix2d A, Vector2d b)
{
	Vector2d x = Vector2d::Zero();
	ColPivHouseholderQR<Matrix2d> qr(A);
	x = qr.solve(b);
	
	return x;
}


double rel_err(Vector2d x_ex, Vector2d sol)
{
	return (x_ex-sol).norm()/x_ex.norm();
}


int main()
{
	Vector2d x_ex;
	x_ex << -1.0, -1.0;
	
	Matrix2d A1;
	A1 << 5.547001962252291e-01, -3.770900990025203e-02,
	8.320502943378437e-01, -9.992887623566787e-01;
	
	Vector2d b1;
	b1 << -5.169911863249772e-01, 1.672384680188350e-01;
	
	Vector2d x1_LU = solutionLU(A1,b1);
	Vector2d x1_QR = solutionQR(A1,b1);
	
	double err1_LU = rel_err(x_ex, x1_LU);
	double err1_QR = rel_err(x_ex, x1_QR);
	
	cout << "L'errore relativo dato dalla fattorizzazione LU del primo sistema è: " << err1_LU << "\n";
	cout << "L'errore relativo dato dalla fattorizzazione QR del primo sistema è: " << err1_QR << "\n";
    
	return 0;
}
