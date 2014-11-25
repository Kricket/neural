package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;

/**
 * A single layer of a CNN.
 */
public abstract class Layer {
	/**
	 * Get the dimensions of the output of this Layer.
	 * @param inputDimension The Dimension of the input of this Layer.
	 * @throws IncompatibleLayerException If this Layer is incompatible with the given input Dimension 
	 */
	abstract public Dimension getOutputDimension(Dimension inputDimension) throws IncompatibleLayerException;
	
	/**
	 * Run the given feature maps forward through this layer.
	 * @param featureMaps The inputs to this layer.
	 * @return The resulting feature maps.
	 */
	abstract public Matrix[] feedForward(Matrix[] featureMaps);
	
	/**
	 * Run backpropagation on this layer.
	 * @param prevZ The un-activated output of the previous layer. This value may be changed!
	 * @param deltas The errors, backpropagated from the next layer.
	 * @return The errors (deltas) for this layer.
	 */
	abstract public Matrix[] backprop(Matrix[] prevZ, Matrix[] deltas);
	
	/**
	 * Calculate the gradients, and add them to our running total.
	 * @param prevActivations The activations of the previous layer.
	 * @param deltas The deltas from the next layer.
	 */
	abstract public void calcGradients(Matrix[] prevActivations, Matrix[] deltas);

	/**
	 * Get the last activation value calculated by this Layer.
	 */
	abstract public Matrix[] lastActivation();
	
	/**
	 * Get the last unactivated output calculated by this layer.
	 */
	abstract public Matrix[] lastZ();
	
	/**
	 * Apply the last-calculated nablas to the weights and biases of this layer.
	 * @param batchSize 
	 * @param eta 
	 * @param regTerm 
	 */
	abstract public void applyGradients(double regTerm, double eta, int batchSize);
	
	/**
	 * Reset the accumulated gradients.
	 */
	abstract public void resetGradients();
	
	/**
	 * The sigma function, for smoothing.
	 * @param z
	 */
	public static double sigma(double z) {
		return 1. / (Math.expm1(-z) + 2.);
	}
	
	/**
	 * Apply the sigma function to every element of the given Matrix. Note that this will CHANGE the matrix!
	 * @return The (modified) input matrix.
	 */
	public static Matrix sigma(Matrix m) {
		for(int i=0; i<m.data.length; i++)
			m.data[i] = sigma(m.data[i]);
		return m;
	}
	
	/**
	 * Derivative of the sigma function.
	 * @param z
	 */
	public static double dSigma(double z) {
		double sigma = sigma(z);
		return sigma * (1 - sigma);
	}
	
	/**
	 * Apply the derivative of the sigma function to every element of the given Matrix. Note that this will CHANGE the matrix!
	 * @return The (modified) input matrix.
	 */
	public static Matrix dSigma(Matrix m) {
		for(int i=0; i<m.data.length; i++)
			m.data[i] = dSigma(m.data[i]);
		return m;
	}
	
	/**
	 * Flatten the given list of matrices into a single column vector.
	 * @param m
	 * @return
	 */
	public static Matrix flatten(Matrix[] m) {
		if(m.length == 1)
			return new Matrix(m[0].data);
		
		int matrixDataLength = m[0].data.length;
		int totalSize = m.length * matrixDataLength;
		double[] flatData = new double[totalSize];
		for(int mi=0; mi<m.length; mi++) {
			System.arraycopy(m[mi].data, 0, flatData, matrixDataLength*mi, matrixDataLength);
		}
		return new Matrix(flatData);
	}
}
