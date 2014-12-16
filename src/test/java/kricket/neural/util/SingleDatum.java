package kricket.neural.util;

/**
 * A very simple datum: a single input value is expected to become a single output value.
 */
public class SingleDatum implements Datum {
	private final Matrix data, answer;
	
	public SingleDatum(double in, double out) {
		data = new Matrix(1,1,new double[]{in});
		answer = new Matrix(1,1,new double[]{out});
	}
	
	@Override
	public Matrix getData() {
		return data;
	}
	
	@Override
	public Matrix getAnswer() {
		return answer;
	}
	
	@Override
	public String toString() {
		return "in: " + data.data[0] + " out: " + answer.data[0];
	}

	@Override
	public Tensor getDataTensor() {
		return new Tensor(data.data);
	}

	@Override
	public Tensor getAnswerTensor() {
		return new Tensor(answer.data);
	}

	@Override
	public int getAnswerClass() {
		return (int) data.data[0];
	}
}
