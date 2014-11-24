package kricket.neural.cnn;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import kricket.neural.util.Datum;
import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
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
}
