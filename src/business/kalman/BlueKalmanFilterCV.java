package business.kalman;

import business.IFilter;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BlueKalmanFilterCV implements IFilter {

    private RealMatrix F, H, Q, R;
    private RealMatrix S, Sp, K;

    private RealMatrix Sa;
    private RealVector xa, xp;
    private RealVector residualMeasurement;
    private double lambda;

    private double var_r, var_theta, var_vr;

    private double lambda1, lambda2, lambda3;
    private RealMatrix blue1, blue2, blue3;

    private double time;

    public BlueKalmanFilterCV(RealVector z, RealVector z_stdDev, double time, double dt) {

        var_r     = z_stdDev.getEntry(0)*z_stdDev.getEntry(0);
        var_theta = z_stdDev.getEntry(1)*z_stdDev.getEntry(1);
        var_vr    = z_stdDev.getEntry(2)*z_stdDev.getEntry(2);

        lambda1 = Math.exp(-var_theta/2);
        lambda2 = (1+ Math.exp(-2*var_theta))/2;
        lambda3 = (1- Math.exp(-2*var_theta))/2;

        blue1 = MatrixUtils.createRealDiagonalMatrix(new double [] {lambda1,lambda1,1});
        blue2 = MatrixUtils.createRealDiagonalMatrix(new double [] {lambda2/lambda1,lambda2/lambda1,1});
        blue3 = MatrixUtils.createRealDiagonalMatrix(new double [] {lambda1,1,lambda1,1});

        // Covariance matrix of state transition noise
        RealMatrix Qx = MatrixUtils.createRealDiagonalMatrix(new double [] {1,1});
        RealMatrix B = MatrixUtils.createRealMatrix(new double [][] {{dt*dt/2,0},{dt,0},{0,dt*dt/2},{0,dt}});
        Q = B.multiply(Qx.multiply(B.transpose()));

        double range =  z.getEntry(0);
        double vr = z.getEntry(2);
        double x = z.getEntry(0)* Math.cos(z.getEntry(1));
        double y = z.getEntry(0)* Math.sin(z.getEntry(1));

        double [] xax = {x/lambda1, x*vr/range, y/lambda1, y*vr/range};
        xa = MatrixUtils.createRealVector(xax);

        double [][] sax = { {var_r,var_r/dt,0,0},
                            {var_r/dt,2*var_r/dt,0,0},
                            {0,0,var_r,var_r/dt},
                            {0,0,var_r/dt,2*var_r/dt}};
        Sa = MatrixUtils.createRealMatrix(sax).scalarMultiply(9);

        this.time = time;
    }

    public RealVector getResidualMeasurement() {
        return residualMeasurement;
    }

    public void setEstimate(RealVector xa) {
        this.xa = xa;
    }

    public RealVector getEstimatedPolar() {
        double range = Math.sqrt(xa.getEntry(0)*xa.getEntry(0) + xa.getEntry(2)*xa.getEntry(2));
        double azimuth = Math.atan2(xa.getEntry(2),xa.getEntry(0));
        double v_r = (xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(2))/range;
        return MatrixUtils.createRealVector(new double [] {range,azimuth,v_r});
    }

    @Override
    public RealVector getEstimate() {
        return xa;
    }

    public RealMatrix getMeasurementModel() {

        double range = Math.sqrt(xa.getEntry(0)*xa.getEntry(0) + xa.getEntry(2)*xa.getEntry(2));
        double v_r = (xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(2))/range;

        double [][] hx = {  {1, 0, 0, 0},
                            {0, 0, 1, 0},
                            {(xa.getEntry(1)-xa.getEntry(0)*v_r/range)/range, xa.getEntry(0)/range, (xa.getEntry(3)-xa.getEntry(2)*v_r/range)/range, xa.getEntry(2)/range}};

        RealMatrix H = MatrixUtils.createRealMatrix(hx);

        return H;
    }

    private RealVector stateToMeasurement(RealVector x) {
        double range = Math.sqrt(x.getEntry(0)*x.getEntry(0) + x.getEntry(2)*x.getEntry(2));
        double azimuth = Math.atan2(x.getEntry(2),x.getEntry(0));
        double v_r = (x.getEntry(1)*x.getEntry(0) + x.getEntry(3)*x.getEntry(2))/range;
        return MatrixUtils.createRealVector(new double [] {range, azimuth, v_r});
    }

    private RealVector stateToConvertedMeasurement(RealVector x) {
        double range = Math.sqrt(x.getEntry(0)*x.getEntry(0) + x.getEntry(2)*x.getEntry(2));
        double azimuth = Math.atan2(x.getEntry(2),x.getEntry(0));
        double v_r = (x.getEntry(1)*x.getEntry(0) + x.getEntry(3)*x.getEntry(2))/range;
        return MatrixUtils.createRealVector(new double [] {x.getEntry(0), x.getEntry(2), v_r});
    }

    private RealVector measureToConvertedMeasurement(RealVector z) {
        double vr = z.getEntry(2);
        double x = z.getEntry(0)* Math.cos(z.getEntry(1));
        double y = z.getEntry(0)* Math.sin(z.getEntry(1));
        return MatrixUtils.createRealVector(new double [] {x, y, vr});
    }

    @Override
    public RealMatrix getResidualCovariance() {
        return S;
    }

    private RealMatrix getR() {
        double alpha  = (var_r)/(xp.getEntry(0)*xp.getEntry(0)+xp.getEntry(2)*xp.getEntry(2));
        double alpha1 = (lambda2-lambda1*lambda1)*(xp.getEntry(0)*xp.getEntry(0))+lambda3*(xp.getEntry(2)*xp.getEntry(2));
        double alpha2 = (lambda2-lambda1*lambda1)*(xp.getEntry(2)*xp.getEntry(2))+lambda3*(xp.getEntry(0)*xp.getEntry(0));
        double alpha4 = (lambda2-lambda3-lambda1*lambda1)*xp.getEntry(0)*xp.getEntry(2);

        double R11 =  lambda3*Sp.getEntry(2,2)  + alpha*(lambda2*xp.getEntry(0)*xp.getEntry(0) + lambda3*xp.getEntry(2)*xp.getEntry(2)) + alpha1;
        double R12 = -lambda3*(Sp.getEntry(0,2) + alpha*xp.getEntry(0)*xp.getEntry(2)) + alpha4;
        double R21 = -lambda3*(Sp.getEntry(2,0) + alpha*xp.getEntry(2)*xp.getEntry(0)) + alpha4;
        double R22 =  lambda3*Sp.getEntry(0,0)  + alpha*(lambda2*xp.getEntry(2)*xp.getEntry(2) + lambda3*xp.getEntry(0)*xp.getEntry(0)) + alpha2;
        double R13 = 0;
        double R23 = 0;
        double R33 = var_vr;

        double [][] rx = {  {R11,R12,R13},
                            {R21,R22,R23},
                            {R13,R23,R33}};

        return MatrixUtils.createRealMatrix(rx);
    }

    public RealVector getPrediction() {
        return xp;
    }

    @Override
    public RealVector predict(double time) {
        double dt = time-this.time;
        this.time = time;

        F     = MatrixUtils.createRealMatrix(new double [][] {{1,dt,0,0},{0,1,0,0},{0,0,1,dt},{0,0,0,1}});
        xp = F.operate(xa);
        Sp = F.multiply(Sa).multiply(F.transpose()).add(Q);

        RealMatrix H = getMeasurementModel();
        S = (blue2.multiply(H).multiply(Sp).multiply(blue1.multiply(H).transpose())).add(getR());    //Computation of the residual covariance

        return xp;
    }

    public RealMatrix getErrorCovariance() {
        return Sa;
    }

    public void setErrorCovariance(RealMatrix Sa) {
        this.Sa = Sa;
    }

    public double getLambda() {
        return lambda;
    }

    @Override
    public RealVector estimate(RealVector z) {
        RealMatrix H = getMeasurementModel();
        RealMatrix invS = MatrixUtils.inverse(S);
        RealVector residualConvertedMeasurement = measureToConvertedMeasurement(z).subtract(blue1.operate(stateToConvertedMeasurement(xp)));

        residualMeasurement = z.subtract(stateToMeasurement(xp));
        lambda = Math.exp(-0.5*residualMeasurement.dotProduct(invS.operate(residualMeasurement)))/ Math.sqrt((new LUDecomposition(S.scalarMultiply(2* Math.PI))).getDeterminant());

        K = Sp.multiply(blue1.multiply(H).transpose()).multiply(invS); //Computation of Kalman Gain
        xa = xp.add(K.operate(residualConvertedMeasurement));             // Computation of the estimate
        Sa = Sp.subtract(K.multiply(H.multiply(Sp)));

        return xa;
    }
}
