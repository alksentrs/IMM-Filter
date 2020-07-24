package business;

import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import util.MyMatrixUtils;

import java.util.Iterator;
import java.util.Vector;

public class IMMFilter implements IFilter{

    Vector<IFilter> filters;

    RealVector modelProbability;
    RealMatrix modelTransitionProbability;

    RealVector xa, xp;
    RealMatrix Sa, S;

    RealVector xpGate;
    RealMatrix SGate;

    public IMMFilter(Vector<IFilter> filters, RealVector modelProbability, RealMatrix modelTransitionProbability) {
        this.modelProbability = modelProbability;
        this.modelTransitionProbability = modelTransitionProbability;
        this.filters = filters;
        mixingEstimates();
    }

    @Override
    public RealMatrix getResidualCovariance() {
        return S;
    }

    @Override
    public RealMatrix getErrorCovariance() {
        return Sa;
    }

    @Override
    public void setErrorCovariance(RealMatrix Sa) {
        this.Sa = Sa;
    }

    @Override
    public RealVector getPredicted() {
        return xp;
    }

    public RealVector predict(double dt) {
        mixingInitialConditions();
        if (filters.size() > 1) {
            double detS = Double.MAX_VALUE;
            RealVector mu_p = modelTransitionProbability.preMultiply(modelProbability);
            int idx = 0;
            Iterator<IFilter> it = filters.iterator();
            while (it.hasNext()) {
                IFilter filter = it.next();
                filter.predict(dt);
                filter.computeResidualCovariance();
                RealVector xpf = filter.getPredicted();
                RealMatrix Sf = filter.getResidualCovariance();
                LUDecomposition luDecomposition = new LUDecomposition(Sf);
                double detSf = luDecomposition.getDeterminant();
                if ( detSf < detS ) {
                    xpGate = xpf;
                    SGate = Sf;
                    detS = detSf;
                }
                RealVector multiplied = xpf.mapMultiply(mu_p.getEntry(idx));
                if (idx == 0) xp = multiplied; else xp = xp.add(multiplied);
                idx++;
            }
        } else {
            IFilter filter = filters.get(0);
            filter.predict(dt);
            xpGate = filter.getPredicted();
            xp = filter.getPredicted();
            filter.computeResidualCovariance();
            SGate = filter.getResidualCovariance();
        }
        return xp;
    }

    private void mixingInitialConditions() {

        if (filters.size()>1) {
            RealVector muCurrent = modelTransitionProbability.preMultiply(modelProbability);

            RealMatrix muTrans = MatrixUtils.createRealMatrix(filters.size(),filters.size());

            // Calc. da mistura das probabilidades de transicao.
            for (int i=0; i<filters.size(); i++) {
                for (int j=0; j<filters.size(); j++) {
                    muTrans.setEntry(i,j, modelTransitionProbability.getEntry(i,j)* modelProbability.getEntry(i)/muCurrent.getEntry(j));
                }
            }

            //System.out.println("mixingInitialConditions muTrans: ");
            MyMatrixUtils.printm(muTrans);

            // Merge of filters initial conditions
            Vector<RealVector> xafMixied = new Vector<>();
            Vector<RealMatrix> SafMixied = new Vector<>();

            for (int j = 0; j < filters.size(); j++) {
                RealVector xa_0 = null;
                RealMatrix Sa_0 = null;
                for (int i = 0; i < filters.size(); i++) {
                    IFilter filter = filters.get(i);
                    RealVector xaf = filter.getEstimated();
                    RealVector multiplied = xaf.mapMultiply(muTrans.getEntry(i,j));
                    if (i == 0) xa_0 = multiplied; else xa_0 = xa_0.add(multiplied);
                }
                for (int i = 0; i < filters.size(); i++) {
                    IFilter filter = filters.get(i);
                    RealVector xaf = filter.getEstimated();
                    RealMatrix Saf = filter.getErrorCovariance();
                    RealVector subtracted = xaf.subtract(xa_0);
                    Saf.add(subtracted.outerProduct(subtracted).scalarMultiply(muTrans.getEntry(i, j)));
                    if (i == 0) Sa_0 = Saf; else Sa_0 = Sa_0.add(Saf);
                }
                xafMixied.add(xa_0);
                SafMixied.add(Sa_0);
            }
            for (int j = 0; j < filters.size(); j++) {
                filters.get(j).setEstimated(xafMixied.get(j));
                filters.get(j).setErrorCovariance(SafMixied.get(j));
            }
        }
    }

    @Override
    public RealVector getResidualMeasurement() {
        return null;
    }

    @Override
    public double getLambda() {
        return 0;
    }

    public RealVector getPredictionGate() {
        return xpGate;
    }

    public RealMatrix getLargestResidualCovariance() {
        return SGate;
    }

    public RealVector getModelProbability() {
        return modelProbability;
    }

    @Override
    public RealVector getEstimated() {
        return xa;
    }

    @Override
    public void setEstimated(RealVector xa) {
        this.xa = xa;
    }

    @Override
    public RealVector estimate(RealVector z) {
        for (int i = 0; i < filters.size(); i++) filters.get(i).estimate(z);
        updateProbabilities();
        mixingPredictions();
        mixingEstimates();
        return xa;
    }

    public RealVector estimateWithPrediction() {
        for (int j = 0; j < filters.size(); j++) filters.get(j).setEstimated(xp);
        return xp;
    }

    private void updateProbabilities() {
        //System.out.println("updateProbabilities()");
        if (filters.size()>1) {
            double sum = 0;
            RealVector muAux = modelProbability.copy();
            //System.out.println(mu);
            //System.out.println(muAux);

            for (int i = 0; i < filters.size(); i++) {
                modelProbability.setEntry(i, modelProbability.getEntry(i) * filters.get(i).getLambda());
                sum = sum + modelProbability.getEntry(i);
            }
            //System.out.println(modelProbability);
            if (sum != 0) {
                for (int i = 0; i < filters.size(); i++) modelProbability.setEntry(i, modelProbability.getEntry(i) / sum);
            } else {
                modelProbability.setEntry(0,muAux.getEntry(0));
                modelProbability.setEntry(1,muAux.getEntry(1));
            }
            //System.out.println(mu);
        }
    }

    @Override
    public void computeResidualCovariance() {
        //S =
    }

    public void mixingPredictions() {
        RealVector mu_p = modelTransitionProbability.preMultiply(modelProbability);

        if (filters.size()>1) {
            for (int i = 0; i < filters.size(); i++) {
                IFilter filter = filters.get(i);
                RealVector xpf = filter.getPredicted();
                RealVector multiplied = xpf.mapMultiply(mu_p.getEntry(i));
                if (i == 0) xp = multiplied; else xp = xp.add(multiplied);
            }
            for (int i = 0; i < filters.size(); i++) {
                IFilter filter = filters.get(i);
                RealVector residueF = filter.getResidualMeasurement();
                RealMatrix Sf = filter.getResidualCovariance();
                RealMatrix added = (Sf.add(residueF.outerProduct(residueF))).scalarMultiply(mu_p.getEntry(i));
                if (i == 0) S = added; else S = S.add(added);
            }
        } else {
            xp = filters.get(0).getPredicted();
            S = filters.get(0).getResidualCovariance();
        }
    }

    private void mixingEstimates() {

        if (filters.size()>1) {
            for (int i = 0; i < filters.size(); i++) {
                IFilter filter = filters.get(i);
                RealVector xaf = filter.getEstimated();
                RealVector multiplied = xaf.mapMultiply(modelProbability.getEntry(i));
                if (i == 0) xa = multiplied; else xa = xa.add(multiplied);
            }
            for (int i = 0; i < filters.size(); i++) {
                IFilter filter = filters.get(i);
                RealVector xaf = filter.getEstimated();
                RealMatrix Saf = filter.getErrorCovariance();
                RealVector subtracted = xa.subtract(xaf);
                RealMatrix added = (Saf.add(subtracted.outerProduct(subtracted))).scalarMultiply(modelProbability.getEntry(i));
                if (i == 0) Sa = added; else Sa = Sa.add(added);
            }
        } else {
            xa = filters.get(0).getEstimated();
            Sa = filters.get(0).getErrorCovariance();
        }
    }
}
