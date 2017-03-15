package arima;

import java.util.ArrayList;
import java.util.Random;
import java.util.Vector;

public class ARIMAModel
{
	double [] originalData = {};
	double [] dataFirDiff = {};
	
	Vector<double []>arimaCoe = new Vector<>();
	
	public ARIMAModel()
	{
		
	}
	
	public ARIMAModel(double [] originalData)
	{
		this.originalData = originalData;
	}
	
	public double [] preFirDiff(double [] preData)		//一阶差分(1)
	{	
		double [] tmpData = new double[preData.length - 1];
		for (int i = 0; i < preData.length - 1; ++i)
		{
			tmpData[i] = preData[i + 1] - preData[i];
		}
		return tmpData;
	}
	
	public double [] preSeasonDiff(double [] preData)		//季节性差分(6, 7)
	{	
		double [] tmpData = new double[preData.length - 7];
		for (int i = 0; i < preData.length - 7; ++i)
		{
			tmpData[i] = preData[i + 7] - preData[i];
		}
		return tmpData;
	}
	
	public double [] preDealDiff(int period)
	{
		if (period >= originalData.length - 1)		// 将6也归为季节性差分
		{
			period = 0;
		}
		switch (period)
		{
		case 0:
			return this.originalData;
		case 1:		 
			this.dataFirDiff = this.preFirDiff(this.originalData);
			return this.dataFirDiff;
		default:	
			return preSeasonDiff(originalData);
		}
	}
	
	public int [] getARIMAModel(int period, ArrayList<int []>notModel, boolean needNot)
	{
		double [] data = this.preDealDiff(period);
		
		double minAIC = Double.MAX_VALUE;
		int [] bestModel = new int[3];
		int type = 0;
		Vector<double []>coe = new Vector<>();
		
		// model产生, 即产生相应的p, q参数
		int len = data.length;
		if (len > 5)
		{
			len = 5;
		}
		int size = ((len + 2) * (len + 1)) / 2 - 1;
		int [][] model = new int[size][2];
		int cnt = 0;
		for (int i = 0; i <= len; ++i)
		{
			for (int j = 0; j <= len - i; ++j)
			{
				if (i == 0 && j == 0)
					continue;
				model[cnt][0] = i;
				model[cnt++][1] = j;
			}
		}

		for (int i = 0; i < model.length; ++i)
		{
			// 控制选择的参数
			boolean token = false;
			if (needNot)
			{
				for (int k = 0; k < notModel.size(); ++k)
				{
					if (model[i][0] == notModel.get(k)[0] && model[i][1] == notModel.get(k)[1])
					{
						token = true;
						break;
					}
				}
			}
			if (token)
			{
				continue;
			}
			
			if (model[i][0] == 0)
			{
				MAModel ma = new MAModel(data, model[i][1]);
				coe = ma.solveCoeOfMA();
				type = 1;
			}
			else if (model[i][1] == 0)
			{
				ARModel ar = new ARModel(data, model[i][0]);
				coe = ar.solveCoeOfAR();
				type = 2;
			}
			else
			{
				ARMAModel arma = new ARMAModel(data, model[i][0], model[i][1]);
				coe = arma.solveCoeOfARMA();
				type = 3;
			}				
			double aic = new ARMAMethod().getModelAIC(coe, data, type);
			// 在求解过程中如果阶数选取过长，可能会出现NAN或者无穷大的情况
			if (Double.isFinite(aic) && !Double.isNaN(aic) && aic < minAIC)
			{
				minAIC = aic;
				bestModel[0] = model[i][0];
				bestModel[1] = model[i][1];
				bestModel[2] = (int)Math.round(minAIC);
				this.arimaCoe = coe;
			}
		}
		return bestModel;
	}
	
	public int aftDeal(int predictValue, int period)
	{
		if (period >= originalData.length)
		{
			period = 0;
		}
		
		switch (period)
		{
		case 0:
			return (int)predictValue;
		case 1:
			return (int)(predictValue + originalData[originalData.length - 1]);
		case 2:
		default:	
			return (int)(predictValue + originalData[originalData.length - 7]);
		}
	}
	
	public int predictValue(int p, int q, int period)
	{
		double [] data = this.preDealDiff(period);
		int n = data.length;
		int predict = 0;
		double tmpAR = 0.0, tmpMA = 0.0;
		double [] errData = new double[q + 1];
		
		Random random = new Random();
		
		if (p == 0)
		{
			double [] maCoe = this.arimaCoe.get(0);
			for(int k = q; k < n; ++k)
			{
				tmpMA = 0;
				for(int i = 1; i <= q; ++i)
				{
					tmpMA += maCoe[i] * errData[i];
				}
				//产生各个时刻的噪声
				for(int j = q; j > 0; --j)
				{
					errData[j] = errData[j - 1];
				}
				errData[0] = random.nextGaussian()*Math.sqrt(maCoe[0]);
			}
			
			predict = (int)(tmpMA); //产生预测
		}
		else if (q == 0)
		{
			double [] arCoe = this.arimaCoe.get(0);
			
			for(int k = p; k < n; ++k)
			{
				tmpAR = 0;
				for(int i = 0; i < p; ++i)
				{
					tmpAR += arCoe[i] * data[k - i - 1];
				}
			}
			predict = (int)(tmpAR);
		}
		else
		{
			double [] arCoe = this.arimaCoe.get(0);
			double [] maCoe = this.arimaCoe.get(1);
			
			for(int k = p; k < n; ++k)
			{
				tmpAR = 0;
				tmpMA = 0;
				for(int i = 0; i < p; ++i)
				{
					tmpAR += arCoe[i] * data[k- i - 1];
				}
				for(int i = 1; i <= q; ++i)
				{
					tmpMA += maCoe[i] * errData[i];
				}
			
				//产生各个时刻的噪声
				for(int j = q; j > 0; --j)
				{
					errData[j] = errData[j-1];
				}
				
				errData[0] = random.nextGaussian() * Math.sqrt(maCoe[0]);
			}
			
			predict = (int)(tmpAR + tmpMA);
		}
		
		return predict;
	}
}
