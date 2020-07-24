package business.kalman;

import business.IFilter;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class BlueKalmanFilterCT implements IFilter {

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

    public BlueKalmanFilterCT(RealVector z, RealVector z_stdDev, double time, double dt) {

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

        double [] b3 = {lambda1,1,lambda1,1};
        blue3 = MatrixUtils.createRealDiagonalMatrix(b3);

        // Covariance matrix of state transition noise
        double [] qx1 = {1,1,1};
        RealMatrix Qx = MatrixUtils.createRealDiagonalMatrix(qx1);
        RealVector W = MatrixUtils.createRealVector(qx1);


        double [][] bx = {{0.5*dt*dt,0,0},{dt,0,0},{0,0.5*dt*dt,0},{0,dt,0},{0,0,1}};
        RealMatrix B = MatrixUtils.createRealMatrix(bx);

        Q = B.multiply(Qx.multiply(B.transpose()));
        //G = B.operate(W);

        double range =  Math.sqrt(z.getEntry(0)*z.getEntry(0)+z.getEntry(1)*z.getEntry(1));

        double [] xax = {z.getEntry(0)/lambda1, z.getEntry(0)*z.getEntry(2)/range, z.getEntry(1)/lambda1, z.getEntry(1)*z.getEntry(2)/range, 1};
        xa = MatrixUtils.createRealVector(xax);

        double r = z_stdDev.getEntry(0)*z_stdDev.getEntry(0);

        double [][] sax = { {r,r/dt,0,0,0},
                            {r/dt,2*r/dt,0,0,0},
                            {0,0,r,r/dt,0},
                            {0,0,r/dt,2*r/dt,0},
                            {0,0,0,0,r}};
        Sa = MatrixUtils.createRealMatrix(sax).scalarMultiply(9);

        this.time = time;
    }

    @Override
    public RealVector getPrediction() {
        return MatrixUtils.createRealVector(new double[] {xp.getEntry(0),xp.getEntry(1),xp.getEntry(2),xp.getEntry(3)});
    }

    @Override
    public RealVector getEstimate() {
        double [] xax = new double[] {xa.getEntry(0),xa.getEntry(1),xa.getEntry(2),xa.getEntry(3)};
        RealVector xa = MatrixUtils.createRealVector(xax);
        return xa;
    }

    @Override
    public void setEstimate(RealVector xa) {
        this.xa.setEntry(0,xa.getEntry(0));
        this.xa.setEntry(1,xa.getEntry(1));

        this.xa.setEntry(2,xa.getEntry(2));
        this.xa.setEntry(3,xa.getEntry(3));
    }

    @Override
    public RealMatrix getErrorCovariance() {
        double [][] Sax = new double[][] {{  Sa.getEntry(0,0),Sa.getEntry(0,1),Sa.getEntry(0,2),Sa.getEntry(0,3)},
                                            {Sa.getEntry(1,0),Sa.getEntry(1,1),Sa.getEntry(1,2),Sa.getEntry(1,3)},
                                            {Sa.getEntry(2,0),Sa.getEntry(2,1),Sa.getEntry(2,2),Sa.getEntry(2,3)},
                                            {Sa.getEntry(3,0),Sa.getEntry(3,1),Sa.getEntry(3,2),Sa.getEntry(3,3)}};
        RealMatrix Sa = MatrixUtils.createRealMatrix(Sax);
        return Sa;
    }

    @Override
    public void setErrorCovariance(RealMatrix Sa) {
        this.Sa.setEntry(0,0,Sa.getEntry(0,0));
        this.Sa.setEntry(0,1,Sa.getEntry(0,1));
        this.Sa.setEntry(0,2,Sa.getEntry(0,2));
        this.Sa.setEntry(0,3,Sa.getEntry(0,3));

        this.Sa.setEntry(1,0,Sa.getEntry(1,0));
        this.Sa.setEntry(1,1,Sa.getEntry(1,1));
        this.Sa.setEntry(1,2,Sa.getEntry(1,2));
        this.Sa.setEntry(1,3,Sa.getEntry(1,3));

        this.Sa.setEntry(2,0,Sa.getEntry(2,0));
        this.Sa.setEntry(2,1,Sa.getEntry(2,1));
        this.Sa.setEntry(2,2,Sa.getEntry(2,2));
        this.Sa.setEntry(2,3,Sa.getEntry(2,3));

        this.Sa.setEntry(3,0,Sa.getEntry(3,0));
        this.Sa.setEntry(3,1,Sa.getEntry(3,1));
        this.Sa.setEntry(3,2,Sa.getEntry(3,2));
        this.Sa.setEntry(3,3,Sa.getEntry(3,3));
    }

    RealMatrix getMeasurementModel() {

        double range = Math.sqrt(xa.getEntry(0)*xa.getEntry(0) + xa.getEntry(2)*xa.getEntry(2));
        double v_r = (xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(2))/range;

        double [][] hx = {  {1, 0, 0, 0, 0},
                            {0, 0, 1, 0, 0},
                            {(xa.getEntry(1)-xa.getEntry(0)*v_r/range)/range, xa.getEntry(0)/range, (xa.getEntry(3)-xa.getEntry(2)*v_r/range)/range, xa.getEntry(2)/range, 0}};

        RealMatrix H = MatrixUtils.createRealMatrix(hx);

        return H;
    }

    @Override
    public RealVector predict(double time) {

        double dt = time-this.time;
        this.time = time;

        double x = xa.getEntry(0);
        double vx = xa.getEntry(1);
        double y = xa.getEntry(2);
        double vy = xa.getEntry(3);
        double w = xa.getEntry(4);

        double sinWdt = Math.sin(dt*w);
        double cosWdt = Math.cos(dt*w);

        double [][] fx = {  {1,        sinWdt/w, 0, (cosWdt - 1)/w, (dt*vx*cosWdt)/w - (vy*(cosWdt - 1))/(w*w) - (vx*sinWdt)/(w*w) - (dt*vy*sinWdt)/w},
                            {0,          cosWdt, 0,        -sinWdt,                                                     - dt*vy*cosWdt - dt*vx*sinWdt},
                            {0, -(cosWdt - 1)/w, 1,       sinWdt/w, (vx*(cosWdt - 1))/(w*w) - (vy*sinWdt)/(w*w) + (dt*vy*cosWdt)/w + (dt*vx*sinWdt)/w},
                            {0,          sinWdt, 0,         cosWdt,                                                       dt*vx*cosWdt - dt*vy*sinWdt},
                            {0,               0, 0,              0,                                                                                 1}};

        F = MatrixUtils.createRealMatrix(fx);

        xp = F.operate(xa);
        Sp = F.multiply(Sa).multiply(F.transpose()).add(Q);

        RealMatrix H = getMeasurementModel();
        S = blue2.multiply(H).multiply(Sp.multiply(blue1.multiply(H).transpose())).add(getR());    //Computation of the residual covariance

        return xp;
    }

    public RealVector stateToMeasurement() {
        double range = Math.sqrt(xp.getEntry(0)*xp.getEntry(0) + xp.getEntry(2)*xp.getEntry(2));
        double v_r = (xp.getEntry(1)*xp.getEntry(0) + xp.getEntry(3)*xp.getEntry(2))/range;

        double [] hx = {xp.getEntry(0), xp.getEntry(2), v_r};
        return MatrixUtils.createRealVector(hx);
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

        double R11 =  lambda3*Sp.getEntry(2,2) + alpha*(lambda2*xp.getEntry(0)*xp.getEntry(0) + lambda3*xp.getEntry(2)*xp.getEntry(2)) + alpha1;
        double R12 = -lambda3*(Sp.getEntry(0,2) + alpha*xp.getEntry(0)*xp.getEntry(2)) + alpha4;
        double R21 = -lambda3*(Sp.getEntry(2,0) + alpha*xp.getEntry(2)*xp.getEntry(0)) + alpha4;
        double R22 =  lambda3*Sp.getEntry(0,0) + alpha*(lambda2*xp.getEntry(2)*xp.getEntry(2) + lambda3*xp.getEntry(0)*xp.getEntry(0)) + alpha2;
        double R13 = 0;
        double R23 = 0;
        double R33 = var_vr;

        double [][] rx = {  {R11,R12,R13},
                {R21,R22,R23},
                {R13,R23,R33}};

        RealMatrix R = MatrixUtils.createRealMatrix(rx);
        return R;
    }

    @Override
    public RealVector estimate(RealVector z) {

        RealMatrix H = getMeasurementModel();

        RealMatrix invS = MatrixUtils.inverse(S);

        K = Sp.multiply(blue1.multiply(H).transpose()).multiply(invS);               //Computation of Kalman Gain

        RealVector zp = stateToMeasurement();

        residualMeasurement = z.subtract(blue1.operate(zp));

        RealVector residualMeasurement2 = z.subtract(zp);
        double G = residualMeasurement2.dotProduct(invS.operate(residualMeasurement2));

        lambda = Math.exp(-0.5* residualMeasurement.dotProduct(invS.operate(residualMeasurement)))/ Math.sqrt((new LUDecomposition(S.scalarMultiply(2* Math.PI))).getDeterminant());

        xa = xp.add(K.operate(residualMeasurement));             // Computation of the stimate
        Sa = Sp.subtract(K.multiply(H.multiply(Sp)));

        return xa;
    }

    public RealVector getResidualMeasurement() {
        return residualMeasurement;
    }

    public double getLambda() {
        return lambda;
    }
}
