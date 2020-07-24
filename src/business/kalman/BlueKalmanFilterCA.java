package business.kalman;

import business.IFilter;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BlueKalmanFilterCA implements IFilter {

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

    public BlueKalmanFilterCA(RealVector z, RealVector z_stdDev, double time, double dt) {

        var_r     = z_stdDev.getEntry(0)*z_stdDev.getEntry(0);
        var_theta = z_stdDev.getEntry(1)*z_stdDev.getEntry(1);
        var_vr    = z_stdDev.getEntry(2)*z_stdDev.getEntry(2);

        lambda1 = Math.exp(-var_theta/2);
        lambda2 = (1+ Math.exp(-2*var_theta))/2;
        lambda3 = (1- Math.exp(-2*var_theta))/2;

        double [] b1 = {lambda1,lambda1,1};
        blue1 = MatrixUtils.createRealDiagonalMatrix(b1);

        double [] b2 = {lambda2/lambda1,lambda2/lambda1,1};
        blue2 = MatrixUtils.createRealDiagonalMatrix(b2);

        double [] b3 = {lambda1,1,1,lambda1,1,1};
        blue3 = MatrixUtils.createRealDiagonalMatrix(b3);

        // Covariance matrix of state transition noise
        double [][] qx = {{1, 0,   0,     0,  0,      0},
                          {0, 1,   0,     0,  0,      0},
                          {0, 0,  dt,     0,  0,      0},
                          {0, 0,   0,     1,  0,      0},
                          {0, 0,   0,     0,  1,      0},
                          {0, 0,   0,     0,  0,     dt}};

        Q = MatrixUtils.createRealMatrix(qx);

        double range =  z.getEntry(0);
        double vr = z.getEntry(2);
        double x = z.getEntry(0)* Math.cos(z.getEntry(1));
        double y = z.getEntry(0)* Math.sin(z.getEntry(1));

        double [] xax = {x/lambda1, x*vr/range, 0, y/lambda1, y*vr/range, 0};
        xa = MatrixUtils.createRealVector(xax);

        double [][] sax = { {var_r,    var_r/dt,       0,     0,        0,        0},
                            {var_r/dt,  2*var_r/dt,  var_r/dt,     0,        0,        0},
                            {0,    var_r/dt, 3*var_r/dt,     0,        0,        0},
                            {0,        0,       0,    var_r,    var_r/dt,        0},
                            {0,        0,       0, var_r/dt,  2*var_r/dt,    var_r/dt},
                            {0,        0,       0,     0,    var_r/dt,  3*var_r/dt}};
        Sa = MatrixUtils.createRealMatrix(sax).scalarMultiply(9);

        this.time = time;

    }

    @Override
    public RealVector getEstimate() {
        double [] xax = new double[] {xa.getEntry(0),xa.getEntry(1),xa.getEntry(3),xa.getEntry(4)};
        RealVector xa = MatrixUtils.createRealVector(xax);
        return xa;
    }

    @Override
    public void setEstimate(RealVector xa) {
        this.xa.setEntry(0,xa.getEntry(0));
        this.xa.setEntry(1,xa.getEntry(1));

        this.xa.setEntry(3,xa.getEntry(2));
        this.xa.setEntry(4,xa.getEntry(3));
    }

    @Override
    public RealVector getPrediction() {
        return MatrixUtils.createRealVector(new double[] {xp.getEntry(0),xp.getEntry(1),xp.getEntry(3),xp.getEntry(4)});
    }

    @Override
    public RealMatrix getErrorCovariance() {

        RealMatrix Sa = MatrixUtils.createRealMatrix(4,4);
        Sa.setEntry(0,0,this.Sa.getEntry(0,0));
        Sa.setEntry(0,1,this.Sa.getEntry(0,1));
        Sa.setEntry(0,2,this.Sa.getEntry(0,3));
        Sa.setEntry(0,3,this.Sa.getEntry(0,4));

        Sa.setEntry(1,0,this.Sa.getEntry(1,0));
        Sa.setEntry(1,1,this.Sa.getEntry(1,1));
        Sa.setEntry(1,2,this.Sa.getEntry(1,3));
        Sa.setEntry(1,3,this.Sa.getEntry(1,4));

        Sa.setEntry(2,0,this.Sa.getEntry(3,0));
        Sa.setEntry(2,1,this.Sa.getEntry(3,1));
        Sa.setEntry(2,2,this.Sa.getEntry(3,3));
        Sa.setEntry(2,3,this.Sa.getEntry(3,4));

        Sa.setEntry(3,0,this.Sa.getEntry(4,0));
        Sa.setEntry(3,1,this.Sa.getEntry(4,1));
        Sa.setEntry(3,2,this.Sa.getEntry(4,3));
        Sa.setEntry(3,3,this.Sa.getEntry(4,4));

        return Sa;
    }

    @Override
    public void setErrorCovariance(RealMatrix Sa) {

        this.Sa.setEntry(0,0,Sa.getEntry(0,0));
        this.Sa.setEntry(0,1,Sa.getEntry(0,1));
        this.Sa.setEntry(0,3,Sa.getEntry(0,2));
        this.Sa.setEntry(0,4,Sa.getEntry(0,3));

        this.Sa.setEntry(1,0,Sa.getEntry(1,0));
        this.Sa.setEntry(1,1,Sa.getEntry(1,1));
        this.Sa.setEntry(1,3,Sa.getEntry(1,2));
        this.Sa.setEntry(1,4,Sa.getEntry(1,3));

        this.Sa.setEntry(3,0,Sa.getEntry(2,0));
        this.Sa.setEntry(3,1,Sa.getEntry(2,1));
        this.Sa.setEntry(3,3,Sa.getEntry(2,2));
        this.Sa.setEntry(3,4,Sa.getEntry(2,3));

        this.Sa.setEntry(4,0,Sa.getEntry(3,0));
        this.Sa.setEntry(4,1,Sa.getEntry(3,1));
        this.Sa.setEntry(4,3,Sa.getEntry(3,2));
        this.Sa.setEntry(4,4,Sa.getEntry(3,3));
    }

    public RealMatrix getMeasurementModel() {

        double range = Math.sqrt(xa.getEntry(0)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(3));
        double v_r = (xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(4)*xa.getEntry(3))/range;

        double [][] hx = {  {1, 0, 0, 0, 0, 0},
                            {0, 0, 0, 1, 0, 0},
                {(xa.getEntry(1)-xa.getEntry(0)*v_r/range)/range, xa.getEntry(0)/range, 0, (xa.getEntry(4)-xa.getEntry(3)*v_r/range)/range, xa.getEntry(3)/range, 0}};

        RealMatrix H = MatrixUtils.createRealMatrix(hx);

        return H;
    }

    public RealVector stateToMeasurement() {
        double range = Math.sqrt(xp.getEntry(0)*xp.getEntry(0) + xp.getEntry(3)*xp.getEntry(3));
        double v_r = (xp.getEntry(1)*xp.getEntry(0) + xp.getEntry(4)*xp.getEntry(3))/range;

        double [] hx = {xp.getEntry(0), xp.getEntry(3), v_r};
        return MatrixUtils.createRealVector(hx);
    }

    @Override
    public RealMatrix getResidualCovariance() {
        return S;
    }

    private RealMatrix getR() {
        double alpha  = (var_r)/(xp.getEntry(0)*xp.getEntry(0)+xp.getEntry(3)*xp.getEntry(3));
        double alpha1 = (lambda2-lambda1*lambda1)*(xp.getEntry(0)*xp.getEntry(0))+lambda3*(xp.getEntry(3)*xp.getEntry(3));
        double alpha2 = (lambda2-lambda1*lambda1)*(xp.getEntry(3)*xp.getEntry(3))+lambda3*(xp.getEntry(0)*xp.getEntry(0));
        double alpha4 = (lambda2-lambda3-lambda1*lambda1)*xp.getEntry(0)*xp.getEntry(3);

        double R11 =  lambda3*Sp.getEntry(3,3) + alpha*(lambda2*xp.getEntry(0)*xp.getEntry(0) + lambda3*xp.getEntry(3)*xp.getEntry(3)) + alpha1;
        double R12 = -lambda3*(Sp.getEntry(0,3) + alpha*xp.getEntry(0)*xp.getEntry(3)) + alpha4;
        double R21 = -lambda3*(Sp.getEntry(3,0) + alpha*xp.getEntry(3)*xp.getEntry(0)) + alpha4;
        double R22 =  lambda3*Sp.getEntry(0,0) + alpha*(lambda2*xp.getEntry(3)*xp.getEntry(3) + lambda3*xp.getEntry(0)*xp.getEntry(0)) + alpha2;
        double R13 = 0;
        double R23 = 0;
        double R33 = var_vr;

        double [][] rx = {  {R11,R12,R13},
                            {R21,R22,R23},
                            {R13,R23,R33}};

        return MatrixUtils.createRealMatrix(rx);
    }

    @Override
    public RealVector estimate(RealVector z) {

        RealMatrix H = getMeasurementModel();
        RealMatrix invS = MatrixUtils.inverse(S);
        residualMeasurement = z.subtract(blue1.operate(stateToMeasurement()));
        lambda = Math.exp(-0.5* residualMeasurement.dotProduct(invS.operate(residualMeasurement)))/ Math.sqrt((new LUDecomposition(S.scalarMultiply(2* Math.PI))).getDeterminant());
        K = Sp.multiply(blue1.multiply(H).transpose()).multiply(invS);               //Computation of Kalman Gain
        xa = xp.add(K.operate(residualMeasurement));             // Computation of the stimate
        Sa = Sp.subtract(K.multiply(H.multiply(Sp)));

        return xa;
    }

    @Override
    public RealVector predict(double time) {

        double dt = time-this.time;
        this.time = time;

        F = MatrixUtils.createRealMatrix(new double [][] {
                {1, dt, dt*dt/2, 0,  0,       0},
                {0, 1,       dt, 0,  0,       0},
                {0, 0,        1, 0,  0,       0},
                {0, 0,        0, 1, dt, dt*dt/2},
                {0, 0,        0, 0,  1,      dt},
                {0, 0,        0, 0,  0,       1}
        });

        xp = F.operate(xa);
        Sp = F.multiply(Sa).multiply(F.transpose()).add(Q);

        RealMatrix H = getMeasurementModel();
        S = blue2.multiply(H).multiply(Sp.multiply(blue1.multiply(H).transpose())).add(getR());    //Computation of the residual covariance

        return xp;
    }

    public RealVector getResidualMeasurement() {
        return residualMeasurement;
    }

    public double getLambda() {
        return lambda;
    }
}
