package arima;

import java.util.Vector;

public class ARMAModel
{
	private double [] data = {};
	private int p;		//AR阶数
	private int q;		//MA阶数
	
	public ARMAModel(double [] data, int p, int q)
	{
		this.data = data;
		this.p = p;
		this.q = q;
	}
	
	/**
	 * 在ARMA模型中，首先根据原始数据求得AR模型的自回归系数(AR系数)
	 * 利用AR系数与原始数据，求解的残差序列，根据残差序列的自协方差最终求得ARMA中MA系数
	 * @return ar, ma
	 */
	public Vector<double []> solveCoeOfARMA()
	{
		Vector<double []>vec = new Vector<>();
		
		//ARMA模型
		double [] armaCoe = new ARMAMethod().computeARMACoe(this.data, this.p, this.q);
		//AR系数
		double [] arCoe = new double[this.p + 1];
		System.arraycopy(armaCoe, 0, arCoe, 0, arCoe.length);
		//MA系数
		double [] maCoe = new double[this.q + 1];
		System.arraycopy(armaCoe, (this.p + 1), maCoe, 0, maCoe.length);
		
		vec.add(arCoe);
		vec.add(maCoe);
		
		return vec;
	}
}
