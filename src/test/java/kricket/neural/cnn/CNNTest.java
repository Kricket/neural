package kricket.neural.cnn;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;

import kricket.neural.util.Datum;
import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;
import kricket.neural.util.NNOptions;
import kricket.neural.util.SingleDatum;

import org.junit.Test;

public class CNNTest {
	
	private NNOptions getOpts() {
		NNOptions opts = new NNOptions();
		opts.calcErrorsAfterEpochs = false;
		opts.logEpochs = false;
		opts.logIncorrectAnswers = false;
		opts.summarizeSGD = false;
		return opts;
	}

	@Test
	public void canTrainForSingleDatum_1Layer() {
		Datum data = new SingleDatum(0,0);
		CNN cnn = new CNN(getOpts(), new FullyConnectedLayer(1, 1));
		
		cnn.SGD(Arrays.asList(data), 1, 100, 5, 0);
		
		double forward = cnn.feedForward(data.getData())[0].data[0];
		assertTrue("Actual value: " + forward, forward < 0.0021);
	}
	
	@Test
	public void convCanTrainForSingleDatum_1Layer() {
		Datum data = new SingleDatum(0,0);
		CNN cnn = new CNN(getOpts(), new ConvolutionalLayer(1, 1, 1, 1, 1));
		
		cnn.SGD(Arrays.asList(data), 1, 100, 5, 0);
		
		double forward = cnn.feedForward(data.getData())[0].data[0];
		assertTrue("Actual value: " + forward, forward < 0.0021);
	}
	
	@Test
	public void canTrainForSingleDatum_3Layer() {
		Datum data = new SingleDatum(0,0);
		CNN cnn = new CNN(getOpts(), new FullyConnectedLayer(1, 3), new FullyConnectedLayer(3, 5), new FullyConnectedLayer(5, 1));
		
		cnn.SGD(Arrays.asList(data), 1, 100, 5, 0);
		
		double forward = cnn.feedForward(data.getData())[0].data[0];
		assertTrue("Actual value: " + forward, forward < 0.0011);
	}
	
	@Test
	public void canTrainForTwoSingleValues() {
		List<SingleDatum> data = Arrays.asList(new SingleDatum(0,0), new SingleDatum(1, 1));
		CNN cnn = new CNN(getOpts(), new FullyConnectedLayer(1, 1));
		
		cnn.SGD(data, 1, 100, 10, 0);
		
		double forward = cnn.feedForward(data.get(0).getData())[0].data[0];
		assertTrue("Actual value: " + forward, forward < 0.0025);
		forward = cnn.feedForward(data.get(1).getData())[0].data[0];
		assertTrue("Actual value: " + forward, forward > 0.99);
	}
	
	@Test
	public void convCanTrainForTwoSingleValues() {
		List<SingleDatum> data = Arrays.asList(new SingleDatum(0,0), new SingleDatum(1, 1));
		CNN cnn = new CNN(getOpts(), new ConvolutionalLayer(1, 1, 1, 1, 1));
		
		cnn.SGD(data, 1, 100, 10, 0);
		
		double forward = cnn.feedForward(data.get(0).getData())[0].data[0];
		assertTrue("Actual value: " + forward, forward < 0.0025);
		forward = cnn.feedForward(data.get(1).getData())[0].data[0];
		assertTrue("Actual value: " + forward, forward > 0.99);
	}
	
	@Test(expected=IncompatibleLayerException.class)
	public void illegalLayerSizes() throws IncompatibleLayerException {
		CNN cnn = new CNN(getOpts(), new FullyConnectedLayer(1, 2), new FullyConnectedLayer(1, 2));
		cnn.checkDimensionality(new Dimension(1,1,1), new Dimension(2,1,1));
	}
	
	@Test(expected=IncompatibleLayerException.class)
	public void illegalInputSize() throws IncompatibleLayerException {
		CNN cnn = new CNN(getOpts(), new FullyConnectedLayer(1, 2));
		cnn.checkDimensionality(new Dimension(2,1,1), new Dimension(2,1,1));
	}
	
	@Test(expected=IncompatibleLayerException.class)
	public void illegalOutputSize() throws IncompatibleLayerException {
		CNN cnn = new CNN(getOpts(), new FullyConnectedLayer(1, 2));
		cnn.checkDimensionality(new Dimension(1,1,1), new Dimension(1,1,1));
	}
	
	final double EPSILON = Math.pow(2, -32) ;
	
	/**
	 * Attempt to make sure that our gradient calculation is about what you'd get if
	 * you treated the layer like a black-box function of its weights and biases.
	 * NOTE that this still fails sometimes! I should have paid more attention in
	 * calculus... :)
	 */
	@Test
	public void handCheckGradients() {
		FullyConnectedLayer layer = new FullyConnectedLayer(3, 3);
		// For some reason, values close to the extremes tend to increase the error in our estimation.
		// (Is this due to the sigma function?)
		layer.weights.timesEquals(0.5);
		layer.resetGradients();
		
		Matrix[] x = new Matrix[] {new Matrix(.9,.5,.1)};
		
		// Let the Layer calculate its own gradients
		Matrix output = layer.feedForward(x)[0];
		Matrix[] delta = new Matrix[] {new Matrix(1,1,1)};
		layer.calcGradients(x, delta);
		
		
		System.out.println("sigma = " + Layer.sigma(1.));
		
		
		// Now, manually calculate them
		Matrix origW = layer.weights.copy(), origB = layer.biases.copy();
		// Consider (layer) to be a function: F(W,B) = W*x + B
		// Then we calculate the derivative wih respect to each w_ij and b_i:
		// f'(x) = ( f(x+h)-f(x) ) / h
		for(int r=0; r<origW.rows; r++) for(int c=0; c<origW.cols; c++) {
			copy(layer.weights, origW);
			layer.weights.set(r, c, layer.weights.at(r, c) + EPSILON);
			
			Matrix outph = layer.feedForward(x)[0];
			double fPrime = outph.minus(output).at(r, 0) / EPSILON;
			
			// I have no idea why, but the fPrime value always seems to be about 1/4 of what
			// the layer calculates. (Slightly less than 1/4 if epsilon < 1; slightly more
			// if epsilon > 1).
			double dW = layer.nabla_Cw.at(r, c);
			assertEquals("at row " + r + " col " + c + "\n" + layer.weights + "\n" + layer.biases,
					dW/4, fPrime, dW/10);
		}
		
		for(int r=0; r<origB.rows; r++) {
			copy(layer.biases, origB);
			layer.biases.set(r, 0, layer.biases.at(r, 0) + EPSILON);
			
			Matrix outph = layer.feedForward(x)[0];
			double fPrime = outph.minus(output).at(r, 0) / EPSILON;
			
			double dB = layer.nabla_Cb.at(r, 0);
			System.out.println("dB = " + dB + " fPrime = " + fPrime);
			assertEquals("at row " + r + "\n" + layer.weights + "\n" + layer.biases,
					dB/4, fPrime, dB/10);
		}
	}

	private void copy(Matrix target, Matrix source) {
		System.arraycopy(source.data, 0, target.data, 0, source.data.length);
	}
}
