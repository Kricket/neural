package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.Matrix;

/**
 * A SigmaLayer is the equivalent of simply applying the sigma function on every entry
 * of every incoming Matrix:
 * <pre>output[i,j,k] = sigma(input[i,j,k])</pre>
 */
public class SigmaLayer implements Layer {
	
	private Matrix[] lastX, lastY;

	/**
	 * The smoothing function.
	 * @param z
	 * @return
	 */
	public static double sigma(double z) {
		return 1. / (Math.expm1(-z) + 2.);
	}
	
	/**
	 * Set s = sigma(m), for each entry of m
	 * @param m
	 * @param s
	 * @return
	 */
	public static void sigma(Matrix m, Matrix s) {
		for(int i=0; i<s.data.length; i++)
			s.data[i] = sigma(m.data[i]);
	}
	
	/**
	 * Derivative of the smoothing function
	 * @param z
	 * @return
	 */
	public static double dSigma(double z) {
		double sigma = sigma(z);
		return sigma * (1 - sigma);
	}
	
	/**
	 * MODIFIES the given matrix by applying the derivative of the sigma function
	 * on each element.
	 * @param m
	 * @return
	 */
	public static Matrix dSigma(Matrix m) {
		for(int i=0; i<m.data.length; i++)
			m.data[i] = dSigma(m.data[i]);
		return m;
	}
	
	@Override
	public Matrix[] feedForward(Matrix[] x) {
		lastX = x;
		for(int i=0; i<x.length; i++) {
			sigma(x[i], lastY[i]);
		}
		return lastY;
	}

	@Override
	public Matrix[] backprop(Matrix[] deltas) {
		if(deltas.length != lastX.length)
			throw new IllegalArgumentException("Expected " + lastX.length + " deltas, but got " + deltas.length);
		if(deltas[0].rows != lastX[0].rows)
			throw new IllegalArgumentException("Expected " + lastX[0].rows + " rows, but got " + deltas[0].rows);
		if(deltas[0].cols != lastX[0].cols)
			throw new IllegalArgumentException("Expected " + lastX[0].cols + " cols, but got " + deltas[0].cols);
		
		for(int d=0; d<deltas.length; d++) {
			deltas[d].dotTimesEquals(dSigma(lastX[d]));
		}
		return deltas;
	}

	@Override
	public void resetGradients() {
		// Nothing to do
	}

	@Override
	public void applyGradients(double regTerm, double scale) {
		// Nothing to do
	}

	@Override
	public Dimension prepare(Dimension inputDimension) {
		lastY = new Matrix[inputDimension.depth];
		for(int i=0; i<lastY.length; i++)
			lastY[i] = new Matrix(inputDimension.rows, inputDimension.columns);
		return inputDimension;
	}

	@Override
	public String toString() {
		return getClass().getSimpleName();
	}
}
