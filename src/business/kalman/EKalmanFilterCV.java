package business.kalman;

import business.IFilter;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class EKalmanFilterCV implements IFilter {

    private RealMatrix F, H, Q, R;
    private RealMatrix S, Sp, K;

    private RealMatrix Sa;
    private RealVector xa, xp;
    private RealVector residualMeasurement;
    private double lambda;
    private double time;

    private double var_r, var_theta, var_vr;

    public EKalmanFilterCV(RealVector z, RealVector z_stdDev, double time, double dt) {

        var_r = z_stdDev.getEntry(0) * z_stdDev.getEntry(0);
        var_theta = z_stdDev.getEntry(1) * z_stdDev.getEntry(1);
        var_vr = z_stdDev.getEntry(2) * z_stdDev.getEntry(2);

        // Covariance matrix of state transition noise
        double [] qx1 = {1,1};
        RealMatrix Qx = MatrixUtils.createRealDiagonalMatrix(qx1);
        double [][] bx = {{0.5*dt*dt,0},{dt,0},{0,0.5*dt*dt},{0,dt}};
        RealMatrix B = MatrixUtils.createRealMatrix(bx);
        Q = B.multiply(Qx.multiply(B.transpose()));

        double range =  z.getEntry(0);
        double vr = z.getEntry(2);
        double x = z.getEntry(0)* Math.cos(z.getEntry(1));
        double y = z.getEntry(0)* Math.sin(z.getEntry(1));

        double [] xax = {x, x*vr/range, y, y*vr/range};
        xa = MatrixUtils.createRealVector(xax);

        double [][] sax = { {var_r,var_r/dt,0,0},
                {var_r/dt,2*var_r/dt,0,0},
                {0,0,var_r,var_r/dt},
                {0,0,var_r/dt,2*var_r/dt}};
        Sa = MatrixUtils.createRealMatrix(sax).scalarMultiply(20);

        R = MatrixUtils.createRealDiagonalMatrix(new double [] {var_r,var_theta,var_vr});

        this.time = time;
    }

    @Override
    public void setEstimate(RealVector xa) {
        this.xa = xa;
    }

    public RealMatrix getMeasurementModel() {

        double range = Math.sqrt(xa.getEntry(0)*xa.getEntry(0) + xa.getEntry(2)*xa.getEntry(2));
        double v_r = (xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(2))/range;

        double [][] hx = {  { xa.getEntry(0)/range, 0, xa.getEntry(2)/range, 0},
                {-xa.getEntry(2)/(Math.pow(xa.getEntry(0),2)*(Math.pow(xa.getEntry(2),2)/ Math.pow(xa.getEntry(0),2) + 1)), 0, 1/(xa.getEntry(0)*(Math.pow(xa.getEntry(2),2)/ Math.pow(xa.getEntry(0),2) + 1)), 0},
                {xa.getEntry(1)/range - (xa.getEntry(0)*(xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(2)))/ Math.pow(range,3), xa.getEntry(0)/range, xa.getEntry(3)/range - (xa.getEntry(2)*(xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(2)))/ Math.pow(range,3), xa.getEntry(2)/range}};

        RealMatrix H = MatrixUtils.createRealMatrix(hx);

        return H;
    }

    public RealVector stateToMeasurement(RealVector x) {
        double range = Math.sqrt(x.getEntry(0)*x.getEntry(0) + x.getEntry(2)*x.getEntry(2));
        double azimuth = Math.atan2(x.getEntry(2), x.getEntry(0));
        double v_r = (x.getEntry(1)*x.getEntry(0) + x.getEntry(3)*x.getEntry(2))/range;

        double [] hx = {range, azimuth, v_r};
        return MatrixUtils.createRealVector(hx);
    }

    public RealVector getEstimatedPolar() {
        double range = Math.sqrt(xa.getEntry(0)*xa.getEntry(0) + xa.getEntry(2)*xa.getEntry(2));
        double azimuth = Math.atan2(xa.getEntry(2),xa.getEntry(0));
        double v_r = (xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(2))/range;
        return MatrixUtils.createRealVector(new double [] {range,azimuth*180/ Math.PI,v_r});
    }

    @Override
    public RealVector getEstimate() {
        double [] xax = new double[] {xa.getEntry(0),xa.getEntry(1),xa.getEntry(2),xa.getEntry(3)};
        RealVector xa = MatrixUtils.createRealVector(xax);
        return xa;
    }

    public RealVector getPrediction() {
        return xp;
    }

    public RealVector getResidualMeasurement() {
        return residualMeasurement;
    }

    public RealMatrix getResidualCovariance() {
        return S;
    }

    public RealVector predict(double time) {

        double dt = time - this.time;
        this.time = time;

        double [][] fx = {};
        F = MatrixUtils.createRealMatrix(new double[][] {
                {1,dt,0, 0},
                {0, 1,0, 0},
                {0, 0,1,dt},
                {0, 0,0, 1}
        });

        xp = F.operate(xa);
        Sp = F.multiply(Sa).multiply(F.transpose()).add(Q);

        H = getMeasurementModel();
        S = H.multiply(Sp).multiply(H.transpose()).add(R);                        //Computation of the residual covariance

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

    public RealVector estimate(RealVector z) {

        residualMeasurement = z.subtract(stateToMeasurement(xp));
        RealMatrix invS = MatrixUtils.inverse(S);
        //System.out.println(residualMeasurement);
        //MyMatrixUtil.printm(S);
        //System.out.println((new LUDecomposition(S)).getDeterminant());
        lambda = Math.exp(-0.5*residualMeasurement.dotProduct(invS.operate(residualMeasurement)))/ Math.sqrt(Math.abs((new LUDecomposition(S.scalarMultiply(2* Math.PI))).getDeterminant()));
        //System.out.println("Lambda: "+lambda);
        K = Sp.multiply(H.transpose()).multiply(invS);          //Computation of Kalmman Gain
        xa = xp.add(K.operate(residualMeasurement));                              // Computation of the estimate
        Sa = Sp.subtract(K.multiply(H.multiply(Sp)));

        return xa;
    }
}
