package kricket.neural.cnn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import kricket.neural.mnist.Image;
import kricket.neural.mnist.Loader;
import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;
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
		opts.summarizeSGD = true;
		return opts;
	}
	
	private void totals(CNN cnn) {
		cnn.getOptions().log.info("\n=====================================\nTotals:");
		cnn.getOptions().log.info("Training images:");
		cnn.calc_error(trainingImages);
		cnn.getOptions().log.info("Test images:");
		cnn.calc_error(testImages);
		System.out.println(cnn);
	}
	
	//@Test
	public void equivalentToNN() throws IncompatibleLayerException {
		CNN cnn = new CNN(getOpts(), new Dimension(Image.HEIGHT, Image.WIDTH, 1),
				new FlatteningLayer(),
				new FullyConnectedLayer(30),
				new SigmaLayer(),
				new FullyConnectedLayer(10));
		cnn.SGD(trainingImages, 10, 3, 0.5, 5);
		
		totals(cnn);
	}

	@Test
	public void simpleConv() throws IncompatibleLayerException {
		CNN cnn = new CNN(getOpts(), new Dimension(Image.HEIGHT, Image.WIDTH, 1),
				new ConvolutionalLayer(8, 3, 3, 2, 2),
				new FlatteningLayer(),
				new FullyConnectedLayer(30),
				new SigmaLayer(),
				new FullyConnectedLayer(10));
		/*
		System.out.println("Augmenting images...");
		List<Image> augmentedTraining = new ArrayList<>(trainingImages);
		for(Image i : trainingImages) {
			augmentedTraining.add(i.rotate(Math.PI/6));
			
			augmentedTraining.add(i.shift(-3, -3));
			augmentedTraining.add(i.shift(3, -3));
			augmentedTraining.add(i.shift(-3, 3));
			augmentedTraining.add(i.shift(3, 3));
			
			augmentedTraining.add(i.rotate(-Math.PI/6));
		}
		System.gc();
		*/
		cnn.SGD(trainingImages, 10, 5, 0.5, 1);
		
		totals(cnn);
		/*
		for(int i=0; i<5; i++) {
			Image img = trainingImages.get(i);
			System.out.println(img);
			Matrix[] result = new ConvolutionalLayer(4, 3, 3, 2, 2).feedForward(new Matrix[] {img.getData()});
			for(int j=0; j<result.length; j++) {
				System.out.println("Map " + j);
				System.out.print(result[j].draw());
			}
		}
		*/
	}
}
