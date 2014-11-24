package kricket.neural.nn;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import kricket.neural.util.NNOptions;
import kricket.neural.util.SingleDatum;

import org.junit.Test;

public class NNTest {
	
	private NNOptions getOpts() {
		NNOptions opts = new NNOptions();
		opts.calcErrorsAfterEpochs = false;
		opts.logEpochs = false;
		opts.logIncorrectAnswers = true;
		opts.summarizeSGD = false;
		return opts;
	}
	
	@Test
	public void canTrainForSingleValue_1Layer() {
		List<SingleDatum> data = Arrays.asList(new SingleDatum(0,0));
		NN nn = new NN(getOpts(), 1,1);
		nn.SGD(data, 1, 100, 10, 0);
		
		double forward = nn.feedForward(data.get(0).getData()).data[0];
		assertTrue("Actual value: " + forward, forward < 0.0101);
	}
	
	@Test
	public void canTrainForSingleValue_3Layer() {
		List<SingleDatum> data = Arrays.asList(new SingleDatum(0,0));
		NN nn = new NN(getOpts(), 1,3,5,1);
		nn.SGD(data, 1, 100, 10, 0);
		
		double forward = nn.feedForward(data.get(0).getData()).data[0];
		assertTrue("Actual value: " + forward, forward < 0.0101);
	}
	
	@Test
	public void canTrainForTwoSingleValues() {
		List<SingleDatum> data = Arrays.asList(new SingleDatum(0,0), new SingleDatum(1, 1));
		NN nn = new NN(getOpts(), 1,1);
		nn.SGD(data, 1, 100, 10, 0);
		
		double forward = nn.feedForward(data.get(0).getData()).data[0];
		assertTrue("Actual value: " + forward, forward < 0.0101);
		forward = nn.feedForward(data.get(1).getData()).data[0];
		assertTrue("Actual value: " + forward, forward > 0.99);
	}
}
