package util;

import jeigen.DenseMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class MyMatrixUtils {

    /* Matrix Exponential */
    static public RealMatrix expm(RealMatrix m) {
        DenseMatrix denseMatrix = new DenseMatrix(m.getData());
        DenseMatrix expDenseMatrix = denseMatrix.mexp();
        RealMatrix expMatrix = MatrixUtils.createRealMatrix(m.getRowDimension(),m.getColumnDimension());
        for (int r=0; r<expMatrix.getRowDimension(); r++) for (int c=0; c<expMatrix.getColumnDimension(); c++) expMatrix.setEntry(r,c,expDenseMatrix.get(r,c));
        return expMatrix;
    }

    /* Matrix Logarithm */
    static public RealMatrix logm(RealMatrix m) {
        DenseMatrix denseMatrix = new DenseMatrix(m.getData());
        DenseMatrix logDenseMatrix = denseMatrix.mlog();
        RealMatrix logMatrix = MatrixUtils.createRealMatrix(m.getRowDimension(),m.getColumnDimension());
        for (int r=0; r<logMatrix.getRowDimension(); r++) for (int c=0; c<logMatrix.getColumnDimension(); c++) logMatrix.setEntry(r,c,logDenseMatrix.get(r,c));
        return logMatrix;
    }

    static public void printm(RealMatrix m) {
        System.out.println(m.getRowDimension()+"x"+m.getColumnDimension());
        for (int r=0; r<m.getRowDimension(); r++) System.out.println(m.getRowVector(r));

    }

    static public void printm(RealVector m) {
        System.out.println(1+"x"+m.getDimension());
        System.out.println(m);

    }
}
