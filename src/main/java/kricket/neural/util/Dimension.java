package kricket.neural.util;

/**
 * The dimension of the input or output of a Layer.
 */
public class Dimension {
	public final int columns, rows, depth;
	public Dimension(int r, int c, int d) {
		columns = c;
		rows = r;
		depth = d;
	}
	
	@Override
	public String toString() {
		return "rows=" + rows + " cols=" + columns + " depth=" + depth;
	}
	
	@Override
	public boolean equals(Object o) {
		if(o instanceof Dimension) {
			Dimension d = (Dimension) o;
			return (d.columns == columns && d.rows == rows && d.depth == depth);
		}
		return false;
	}
}
