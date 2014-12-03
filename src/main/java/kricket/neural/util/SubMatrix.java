package kricket.neural.util;

public class SubMatrix extends Matrix {
	private final int rowOff, colOff;
	private final Matrix parent;
	SubMatrix(Matrix parent, int rows, int cols, int rowOffset, int colOffset) {
		super(rows, cols, parent.data);
		this.parent = parent;
		rowOff = rowOffset;
		colOff = colOffset;
	}
	
	@Override
	public double at(int r, int c) {
		return parent.at(r+rowOff, c+colOff);
	}
	
	@Override
	public void set(int r, int c, double value) {
		parent.set(r+rowOff, c+colOff, value);
	}
}
