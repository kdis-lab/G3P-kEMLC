package gpemlc;

import java.util.Arrays;

public class Prediction {
	
	public int nInstances;
	
	public int nLabels;
	
	public byte[][] bip;
	
	public Prediction() {
		nInstances = -1;
		nLabels = -1;
		bip = null;
	}
	
	public Prediction(int nInstances, int nLabels) {
		this.nInstances = nInstances;
		this.nLabels = nLabels;
		this.bip = new byte[nInstances][nLabels];
	}
	
	public void addPrediction(Prediction other) {
		if(this.nInstances != other.nInstances || this.nLabels != other.nLabels) {
			System.out.println("The number of instances or labels is not the same in both predictions.");
			System.exit(-1);
		}
		
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				this.bip[i][j] += other.bip[i][j];
			}
		}
	}
	
	public void divideAndThresholdPrediction(int nPreds, double threshold) {
		for(int i=0; i<nInstances; i++) {
			for(int j=0; j<nLabels; j++) {
				if((this.bip[i][j]*1.0)/nPreds >= threshold) {
					this.bip[i][j] = 1;
				}
				else {
					this.bip[i][j] = 0;
				}
			}
		}
	}
	
	@Override
	public String toString() {
		String s = "";
		
		for(int i=0; i<nInstances; i++) {
			s += Arrays.toString(this.bip[i]) + "\n";
		}
	
		return s;
	}
	
}
