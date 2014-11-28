package kricket.neural.cnn;

import java.util.List;

import kricket.neural.NNBase;
import kricket.neural.util.Datum;
import kricket.neural.util.Dimension;
import kricket.neural.util.IncompatibleLayerException;
import kricket.neural.util.Matrix;
import kricket.neural.util.NNOptions;

public class CNN extends NNBase {

	/**
	 * The layers of this CNN (not counting the input layer).
	 */
	private final Layer[] layers;
	
	public CNN(NNOptions opts, Layer ...layers) {
		super(opts);
		
		if(layers.length < 1)
			throw new IllegalArgumentException("No layers given!");
		
		this.layers = layers;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for(Layer l:layers) {
			sb.append(l);
			sb.append("\n");
		}
		return sb.toString();
	}
	
	/**
	 * Shortcut for {@link #feedForward(Matrix[])}
	 */
	public Matrix[] feedForward(Matrix input) {
		return feedForward(new Matrix[] {input});
	}
	
	/**
	 * Get the result of running the given feature maps through this CNN.
	 * @param featureMaps The initial features.
	 * @return
	 */
	public Matrix[] feedForward(Matrix[] featureMaps) {
		Matrix[] current = featureMaps;
		for(Layer layer : layers) {
			current = layer.feedForward(current);
		}
		return current;
	}
	
	@Override
	protected void runBatch(List<? extends Datum> batch, double regTerm, double eta) {
		for(Layer layer : layers)
			layer.resetGradients();
		
		for(Datum dat : batch)
			backprop(dat.getData(), dat.getAnswer());
		
		for(Layer layer : layers)
			layer.applyGradients(regTerm, eta, batch.size());
	}


	private void backprop(Matrix x, Matrix y) {
		Matrix[] forward = feedForward(x);
		Matrix[] delta = new Matrix[] {forward[0].minus(y)};
		// This delta represents the cross-entropy cost function.
		// To use quadratic cost, you must also do (for each delta and Z):
		//   delta.dotTimesEquals(dSigma(lastLayer.lastZ()))
		
		for(int i=layers.length-1; i>0; i--) {
			layers[i].calcGradients(layers[i-1].lastActivation(), delta);
			delta = layers[i].backprop(layers[i-1].lastZ(), delta);
		}
		
		layers[0].calcGradients(new Matrix[]{x}, delta);
	}

	@Override
	public double calc_error(List<? extends Datum> data) {
		double numCorrect = 0;
		for(Datum dat : data) {
			Matrix[] result = feedForward(dat.getData());
			if(isCorrect(result[0], dat.getAnswer())) {
				numCorrect++;
			} else if(options.logIncorrectAnswers) {
				options.log.info("Got this one wrong:\n" + dat);
				for(double d : result[0].data)
					options.log.info(String.format("%.3f", d));
			}
		}
		
		double correct = numCorrect / data.size();
		options.log.info(String.format("-------------------> Percent correct: %.3f", correct*100));
		return correct;
	}
	
	public void checkDimensionality(Dimension input, Dimension output) throws IncompatibleLayerException {
		for(Layer layer : layers) {
			input = layer.getOutputDimension(input);
		}
		
		if(!input.equals(output)) {
			throw new IncompatibleLayerException(output, layers[layers.length-1]);
		}
	}

}
