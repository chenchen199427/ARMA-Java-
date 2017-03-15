package arima;

import java.util.Vector;

public class ARModel
{
	private double [] data;
	private int p;
	
	public ARModel(double [] data, int p)
	{
		this.data = data;
		this.p = p;
	}
	
	public Vector<double []> solveCoeOfAR()
	{
		Vector<double []>vec = new Vector<>();
		double [] arCoe = new ARMAMethod().computeARCoe(this.data, this.p);
		
		vec.add(arCoe);
		
		return vec;
	}
}
