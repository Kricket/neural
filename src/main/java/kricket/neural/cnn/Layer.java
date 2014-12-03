package kricket.neural.cnn;

import kricket.neural.util.Matrix;

/**
 * A single Layer of a CNN.
 */
public interface Layer {
	/**
	 * Run the given feature maps forward through this layer. The given input
	 * will be stored, in preparation for backpropagation.
	 * @param x The inputs to this layer.
	 * @return The resulting feature maps.
	 */
	Matrix[] feedForward(Matrix[] x);
	
	/**
	 * Calculate the gradient of the parameters of this Layer with respect to
	 * the given errors, and get the errors for the prior Layer. Adds the gradient
	 * to a running total, but does not apply it yet!
	 * @param deltas The errors from the next Layer.
	 * @return The errors for the prior Layer.
	 */
	Matrix[] backprop(Matrix[] deltas);

	/**
	 * Apply the calculated gradients to the parameters of this Layer.
	 * @param regTerm Regularization term - applied only if non-zero!
	 * @param scale Scale factor for the gradients.
	 */
	void applyGradients(double regTerm, double scale);
	
	/**
	 * Reset the calculated gradients to 0.
	 */
	void resetGradients();
}
