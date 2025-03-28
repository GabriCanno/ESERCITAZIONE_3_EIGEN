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
	// PRIMO SISTEMA
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
	cout << "L'errore relativo dato dalla fattorizzazione QR del primo sistema è: " << err1_QR << "\n" << "\n";
    
	
	//SECONDO SISTEMA
	Matrix2d A2;
	A2 << 5.547001962252291e-01, -5.540607316466765e-01,
	8.320502943378437e-01, -8.324762492991313e-01;
	
	Vector2d b2;
	b2 <<  -6.394645785530173e-04, 4.259549612877223e-04;
	
	Vector2d x2_LU = solutionLU(A2,b2);
	Vector2d x2_QR = solutionQR(A2,b2);
	
	double err2_LU = rel_err(x_ex, x2_LU);
	double err2_QR = rel_err(x_ex, x2_QR);
	
	cout << "L'errore relativo dato dalla fattorizzazione LU del secondo sistema è: " << err2_LU << "\n";
	cout << "L'errore relativo dato dalla fattorizzazione QR del secondo sistema è: " << err2_QR << "\n" << "\n";
    
	
	//TERZO SISTEMA
	Matrix2d A3;
	A3 << 5.547001962252291e-01, -5.547001955851905e-01,
	8.320502943378437e-01, -8.320502947645361e-01;
	
	Vector2d b3;
	b3 << -6.400391328043042e-10, 4.266924591433963e-10;
	
	Vector2d x3_LU = solutionLU(A3,b3);
	Vector2d x3_QR = solutionQR(A3,b3);
	
	double err3_LU = rel_err(x_ex, x3_LU);
	double err3_QR = rel_err(x_ex, x3_QR);
	
	cout << "L'errore relativo dato dalla fattorizzazione LU del terzo sistema è: " << err3_LU << "\n";
	cout << "L'errore relativo dato dalla fattorizzazione QR del terzo sistema è: " << err3_QR << "\n" << "\n";
	
	return 0;
}
