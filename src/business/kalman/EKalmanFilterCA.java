package business.kalman;

import business.IFilter;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class EKalmanFilterCA implements IFilter {

    private RealMatrix F, H, Q, R;
    private RealMatrix S, Sp, K;

    private RealMatrix Sa;
    private RealVector xa, xp;
    private RealVector residualMeasurement;
    private double lambda;
    private double time;

    private double var_r, var_theta, var_vr;

    public EKalmanFilterCA(RealVector z, RealVector z_stdDev, double time, double dt) {

        var_r = z_stdDev.getEntry(0) * z_stdDev.getEntry(0);
        var_theta = z_stdDev.getEntry(1) * z_stdDev.getEntry(1);
        var_vr = z_stdDev.getEntry(2) * z_stdDev.getEntry(2);

        // Covariance matrix of state transition noise
        RealMatrix Qx = MatrixUtils.createRealDiagonalMatrix(new double [] {1,1});
        RealMatrix B = MatrixUtils.createRealMatrix(new double [][] {{0.5*dt*dt,0},{dt,0},{1,0},{0,0.5*dt*dt},{0,dt},{0,1}});
        Q = B.multiply(Qx.multiply(B.transpose()));

        double range =  z.getEntry(0);
        double vr = z.getEntry(2);
        double x = z.getEntry(0)* Math.cos(z.getEntry(1));
        double y = z.getEntry(0)* Math.sin(z.getEntry(1));

        double [] xax = {x, x*vr/range, 0, y, y*vr/range, 0};
        xa = MatrixUtils.createRealVector(xax);

        double [][] sax = { {var_r,      var_r/dt,   2*var_r/dt, 0,          0,          0},
                            {var_r/dt,   2*var_r/dt, 0,          0,          0,          0},
                            {2*var_r/dt, 0,          1,          0,          0,          0},
                            {0,          0,          0,          var_r,      var_r/dt,   2*var_r/dt},
                            {0,          0,          0,          var_r/dt,   2*var_r/dt, 0},
                            {0,          0,          0,          2*var_r/dt, 0,          1}};
        Sa = MatrixUtils.createRealMatrix(sax).scalarMultiply(20);

        R = MatrixUtils.createRealDiagonalMatrix(new double [] {var_r,var_theta,var_vr});
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

        double [][] hx = {  { xa.getEntry(0)/range, 0, 0, xa.getEntry(3)/range, 0, 0},
                {-xa.getEntry(3)/(Math.pow(xa.getEntry(0),2)*(Math.pow(xa.getEntry(3),2)/ Math.pow(xa.getEntry(0),2) + 1)), 0, 0, 1/(xa.getEntry(0)*(Math.pow(xa.getEntry(3),2)/ Math.pow(xa.getEntry(0),2) + 1)), 0, 0},
                {xa.getEntry(1)/range - (xa.getEntry(0)*(xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(4)*xa.getEntry(3)))/ Math.pow(range,3), xa.getEntry(0)/range, 0, xa.getEntry(4)/range - (xa.getEntry(3)*(xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(4)*xa.getEntry(3)))/ Math.pow(range,3), xa.getEntry(3)/range, 0}};

        RealMatrix H = MatrixUtils.createRealMatrix(hx);
        return H;
    }

    public RealVector stateToMeasurement(RealVector x) {
        double range = Math.sqrt(x.getEntry(0)*x.getEntry(0) + x.getEntry(3)*x.getEntry(3));
        double azimuth = Math.atan2(x.getEntry(3), x.getEntry(0));
        double v_r = (x.getEntry(1)*x.getEntry(0) + x.getEntry(4)*x.getEntry(3))/range;

        double [] hx = {range, azimuth, v_r};
        return MatrixUtils.createRealVector(hx);
    }

    public RealVector getEstimatedPolar() {
        double range = Math.sqrt(xa.getEntry(0)*xa.getEntry(0) + xa.getEntry(3)*xa.getEntry(3));
        double azimuth = Math.atan2(xa.getEntry(3),xa.getEntry(0));
        double v_r = (xa.getEntry(1)*xa.getEntry(0) + xa.getEntry(4)*xa.getEntry(3))/range;
        return MatrixUtils.createRealVector(new double [] {range,azimuth*180/ Math.PI,v_r});
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

        F = MatrixUtils.createRealMatrix(new double [][] {
                {1, dt, dt*dt/2, 0,  0,       0},
                {0,  1,      dt, 0,  0,       0},
                {0,  0,       1, 0,  0,       0},
                {0,  0,       0, 1, dt, dt*dt/2},
                {0,  0,       0, 0,  1,      dt},
                {0,  0,       0, 0,  0,       1}
        });

        xp = F.operate(xa);
        Sp = F.multiply(Sa).multiply(F.transpose()).add(Q);

        H = getMeasurementModel();
        S = H.multiply(Sp).multiply(H.transpose()).add(R);                        //Computation of the residual covariance

        return xp;
    }

    public double getLambda() {
        return lambda;
    }

    public RealVector estimate(RealVector z) {

        residualMeasurement = z.subtract(stateToMeasurement(xp));
        RealMatrix invS = MatrixUtils.inverse(S);
        //MyMatrixUtil.printm(S);
        //System.out.println(Math.abs((new LUDecomposition(S)).getDeterminant()));
        lambda = Math.exp(-0.5*residualMeasurement.dotProduct(invS.operate(residualMeasurement)))/ Math.sqrt(Math.abs((new LUDecomposition(S.scalarMultiply(2* Math.PI))).getDeterminant()));
        //System.out.println("Lambda: "+lambda);
        K = Sp.multiply(H.transpose()).multiply(invS);          //Computation of Kalmman Gain
        xa = xp.add(K.operate(residualMeasurement));                              // Computation of the estimate
        Sa = Sp.subtract(K.multiply(H.multiply(Sp)));

        return xa;
    }
}
