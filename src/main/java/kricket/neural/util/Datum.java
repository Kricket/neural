package kricket.neural.util;


public interface Datum {

	/**
	 * The input data.
	 * @return
	 */
	Matrix getData();
	/**
	 * The expected answer.
	 * @return
	 */
	Matrix getAnswer();
}
