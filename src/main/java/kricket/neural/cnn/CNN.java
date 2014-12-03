package kricket.neural.cnn;

import java.util.List;

import kricket.neural.NNBase;
import kricket.neural.util.Datum;
import kricket.neural.util.Matrix;
import kricket.neural.util.NNOptions;

public class CNN extends NNBase {

	/**
	 * The layers of this CNN (not counting the input layer).
	 */
	private final Layer[] layers;
	
	/**
	 * Create a new network.
	 * <p><b>Attention: </b>an EXTRA SigmaLayer will be added to the end!
	 * @param opts
	 * @param layers
	 */
	public CNN(NNOptions opts, Layer ...layers) {
		super(opts);
		this.layers = new Layer[layers.length+1];
		for(int i=0; i<layers.length; i++)
			this.layers[i] = layers[i];
		this.layers[layers.length] = new SigmaLayer();
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
	 * Get the result of running the given feature maps through this CNN.
	 * @param x The initial feature maps.
	 * @return
	 */
	public Matrix[] feedForward(Matrix[] x) {
		for(Layer layer : layers) {
			x = layer.feedForward(x);
		}
		return x;
	}
	
	/**
	 * Shortcut for {@link #feedForward(Matrix[])}
	 * @param x
	 * @return
	 */
	public Matrix[] feedForward(Matrix x) {
		return feedForward(new Matrix[]{x});
	}

	@Override
	protected void runBatch(List<? extends Datum> batch, double regTerm, double eta) {
		for(Layer layer : layers)
			layer.resetGradients();
		
		for(Datum dat : batch)
			backprop(dat.getData(), dat.getAnswer());
		
		for(Layer layer : layers)
			layer.applyGradients(regTerm, eta/batch.size());
	}

	/**
	 * Run the backpropagation algorithm.
	 * @param x
	 * @param y
	 */
	private void backprop(Matrix x, Matrix y) {
		Matrix[] forward = feedForward(x);
		Matrix[] deltas = new Matrix[] {forward[0].minus(y)};
		
		// The cross-entropy cost function basically boils down to not running
		// backpropagation on the last (sigmoid) layer. If we instead wanted
		// to use quadratic cost, we would simply include the last layer in the
		// following loop.
		for(int i=layers.length-2; i>=0; i--) {
			deltas = layers[i].backprop(deltas);
		}
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
}
