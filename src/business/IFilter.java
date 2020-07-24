package business;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public interface IFilter {

    RealMatrix getResidualCovariance();

    RealMatrix getErrorCovariance();

    void setErrorCovariance(RealMatrix Sa);

    RealVector getEstimated();

    void setEstimated(RealVector xa);

    RealVector getPredicted();

    RealVector getResidualMeasurement();

    double getLambda();

    RealVector predict(double dt);

    void computeResidualCovariance();

    RealVector estimate(RealVector z);
}
