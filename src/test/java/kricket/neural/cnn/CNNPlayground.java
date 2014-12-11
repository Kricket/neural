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
import kricket.neural.util.Tensor;

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
				new ConvolutionalLayer(4, 3, 3, 2, 2),
				new MaxPoolingLayer(),
				new FlatteningLayer(),
				new SigmaLayer(),
				new FullyConnectedLayer(30, 0.5),
				new SigmaLayer(),
				new FullyConnectedLayer(10)
		);
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
			Tensor result = mpLayer.feedForward(new ConvolutionalLayer(8, 3, 3, 2, 2).feedForward(img.getDataTensor()));
			for(int j=0; j<result.depth; j++) {
				System.out.println("Map " + j);
				System.out.print(result.draw(j));
			}
		}
		*/
	}
	
	//@Test
	public void gradientMagnitudes() throws IncompatibleLayerException {
		FullyConnectedLayer[] layers = new FullyConnectedLayer[] {
				new FullyConnectedLayer(30),
				new FullyConnectedLayer(30),
				new FullyConnectedLayer(30),
				new FullyConnectedLayer(30),
				new FullyConnectedLayer(30),
				new FullyConnectedLayer(30),
				new FullyConnectedLayer(30),
				new FullyConnectedLayer(10)
		};
		CNN cnn = new CNN(getOpts(), new Dimension(Image.HEIGHT, Image.WIDTH, 1),
				new FlatteningLayer(),
				layers[0],
				new SigmaLayer(),
				layers[1],
				new SigmaLayer(),
				layers[2],
				new SigmaLayer(),
				layers[3],
				new SigmaLayer(),
				layers[4],
				new SigmaLayer(),
				layers[5],
				new SigmaLayer(),
				layers[6],
				new SigmaLayer(),
				layers[7]
		);
		cnn.SGD(testImages.subList(0, 1), 1, 5, .5, 0);
		System.out.println("Norms:");
		for(int i=0; i<layers.length; i++) {
			System.out.println("Layer " + i + ": dW = " + layers[i].dW.norm() + " dB = " + layers[i].dB.norm());
		}
	}
	
	//@Test
	public void tryMomentum() throws IncompatibleLayerException {
		CNN cnn = new CNN(getOpts(), new Dimension(Image.HEIGHT, Image.WIDTH, 1),
				new FlatteningLayer(),
				new FullyConnectedLayer(100, .5),
				new SigmaLayer(),
				new FullyConnectedLayer(50, .25),
				new SigmaLayer(),
				new FullyConnectedLayer(30),
				new SigmaLayer(),
				new FullyConnectedLayer(10));
		cnn.SGD(trainingImages, 10, 3, 0.5, 5);
		
		totals(cnn);
	}
	
	//@Test
	public void funWithAvg() {
		Matrix[] avg = new Matrix[10];
		int[] totals = new int[10];
		double[] minDist = new double[10], maxDist = new double[10];
		Image[] mins = new Image[10], maxs = new Image[10];
		
		for(int i=0; i<avg.length; i++) {
			avg[i] = new Matrix(Image.HEIGHT, Image.WIDTH);
			minDist[i] = Double.MAX_VALUE;
		}
		
		for(Image i : testImages) {
			totals[i.getAnswerAsByte()]++;
			avg[i.getAnswerAsByte()].plusEquals(i.getData());
		}
		
		for(int i=0; i<avg.length; i++) {
			avg[i].timesEquals(1. / totals[i]);
		}
		
		for(Image i : testImages) {
			byte idx = i.getAnswerAsByte();
			double dist = avg[idx].minus(i.getData()).norm();
			if(dist < minDist[idx]) {
				mins[idx] = i;
				minDist[idx] = dist;
			}
			
			if(dist > maxDist[idx]) {
				maxs[idx] = i;
				maxDist[idx] = dist;
			}
		}
		
		for(int i=0; i<avg.length; i++) {
			System.out.println("closest for " + i);
			System.out.println(mins[i].getData().draw());
			System.out.println("farthest for " + i);
			System.out.println(maxs[i].getData().draw());
			
		}
	}
}
