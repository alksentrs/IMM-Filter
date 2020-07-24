package business;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public interface IFilter {

    RealMatrix getResidualCovariance();

    RealMatrix getErrorCovariance();

    void setErrorCovariance(RealMatrix Sa);

    RealVector getEstimate();

    void setEstimate(RealVector xa);

    RealVector getPrediction();

    RealVector getResidualMeasurement();

    double getLambda();

    RealVector predict(double time);

    RealVector estimate(RealVector z);
}
