package kricket.neural.cnn;

import kricket.neural.util.Matrix;

/**
 * A SigmaLayer is the equivalent of simply applying the sigma function on every entry
 * of every incoming Matrix:
 * <pre>output[i,j,k] = sigma(input[i,j,k])</pre>
 */
public class SigmaLayer implements Layer {
	
	private Matrix[] lastX;

	/**
	 * The smoothing function.
	 * @param z
	 * @return
	 */
	public static double sigma(double z) {
		return 1. / (Math.expm1(-z) + 2.);
	}
	
	/**
	 * Get a copy of the given matrix, with sigma applied to each entry
	 * @param m
	 * @return
	 */
	public static Matrix sigma(Matrix m) {
		Matrix s = new Matrix(m.rows, m.cols);
		for(int i=0; i<s.data.length; i++)
			s.data[i] = sigma(m.data[i]);
		return s;
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
	 * TODO: probably don't need to copy the matrix here
	 * @param m
	 * @return
	 */
	public static Matrix dSigma(Matrix m) {
		Matrix s = new Matrix(m.rows, m.cols);
		for(int i=0; i<s.data.length; i++)
			s.data[i] = dSigma(m.data[i]);
		return s;
	}
	
	@Override
	public Matrix[] feedForward(Matrix[] x) {
		lastX = x;
		Matrix[] y = new Matrix[x.length];
		for(int i=0; i<x.length; i++) {
			y[i] = sigma(x[i]);
		}
		return y;
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

}
