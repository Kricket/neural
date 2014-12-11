package kricket.neural.cnn;

import kricket.neural.util.Dimension;
import kricket.neural.util.Tensor;

/**
 * A SigmaLayer is the equivalent of simply applying the sigma function on every entry
 * of every incoming Matrix:
 * <pre>output[i,j,k] = sigma(input[i,j,k])</pre>
 */
public class SigmaLayer implements Layer {
	
	private Tensor lastX, lastY;

	/**
	 * The smoothing function.
	 * @param z
	 * @return
	 */
	public static double sigma(double z) {
		return 1. / (Math.expm1(-z) + 2.);
	}
	
	/**
	 * Set s = sigma(m), for each entry of m. This will MODIFY s!
	 * @param m
	 * @param s
	 * @return
	 */
	public static void sigma(Tensor m, Tensor s) {
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
	public static Tensor dSigma(Tensor m) {
		for(int i=0; i<m.data.length; i++)
			m.data[i] = dSigma(m.data[i]);
		return m;
	}
	
	@Override
	public Tensor feedForward(Tensor x) {
		lastX = x;
		sigma(x, lastY);
		return lastY;
	}

	@Override
	public Tensor backprop(Tensor deltas) {
		if(!deltas.getDimension().equals(lastX.getDimension()))
			throw new IllegalArgumentException();
		
		deltas.dotTimesEquals(dSigma(lastX));
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
		lastY = new Tensor(inputDimension);
		return inputDimension;
	}

	@Override
	public String toString() {
		return getClass().getSimpleName();
	}
}
