package kricket.neural.cnn;

import java.io.IOException;
import java.util.List;

import kricket.neural.mnist.Image;
import kricket.neural.mnist.Loader;
import kricket.neural.util.NNOptions;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class CNNPlayground {
	final static int IMGSIZE = Image.HEIGHT*Image.WIDTH;
	List<Image> trainingImages, testImages;
	long startTime;
	
	@Before
	public void loadImages() throws IOException {
		trainingImages = Loader.loadImages(Loader.IMAGES_TRAINING_FILE, Loader.LABELS_TRAINING_FILE);
		testImages = Loader.loadImages(Loader.IMAGES_10K_FILE, Loader.LABELS_10K_FILE);
		startTime = System.currentTimeMillis();
	}
	
	@After
	public void logTime() {
		System.out.println(String.format("Total time: %.3fs", (System.currentTimeMillis() - startTime)*0.001));
	}

	private NNOptions getOpts() {
		NNOptions opts = new NNOptions();
		opts.calcErrorsAfterEpochs = true;
		opts.logDropout = true;
		opts.logEpochs = true;
		opts.logIncorrectAnswers = false;
		opts.summarizeSGD = false;
		return opts;
	}
	
	private void totals(CNN cnn) {
		cnn.getOptions().log.info("\n=====================================\nTotals:");
		cnn.getOptions().log.info("Training images:");
		cnn.calc_error(trainingImages);
		cnn.getOptions().log.info("Test images:");
		cnn.calc_error(testImages);
	}
	
	//@Test
	public void basic() {
		CNN cnn = new CNN(getOpts(), new FullyConnectedLayer(IMGSIZE, 30), new FullyConnectedLayer(30, 10));
		cnn.SGD(trainingImages, 10, 3, 0.5, 5);
		
		totals(cnn);
	}

	@Test
	public void conv() {
		CNN cnn = new CNN(getOpts(), new ConvolutionalLayer(1, 3, 3, 2, 2), new FullyConnectedLayer(169, 30), new FullyConnectedLayer(30, 10));
		cnn.SGD(trainingImages, 10, 10, 0.5, 5);
		
		totals(cnn);
	}
}
