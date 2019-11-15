package gpemlc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.IFitness;
import net.sf.jclec.IIndividual;
import net.sf.jclec.base.AbstractEvaluator;
import net.sf.jclec.base.AbstractParallelEvaluator;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.fitness.ValueFitnessComparator;
import net.sf.jclec.stringtree.StringTreeIndividual;

/**
 * Class implementing the evaluator for individuals (MultipListIndividuals).
 * Each individual is transformed to the corresponding multi-label classifier and then evaluated.
 * 
 * @author Jose M. Moyano
 *
 */
public class Evaluator extends AbstractEvaluator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = 3488129084072488337L;

	/**
	 *  Indicates if fitness is to maximize 
	 */
	private boolean maximize = true;
	
	/**
	 *  Fitness comparator 
	 */
	private Comparator<IFitness> COMPARATOR = new ValueFitnessComparator(!maximize);
	
	/**
	 * Full training dataset
	 */
	MultiLabelInstances fullTrainData;
	
	Utils utils = new Utils();
	
	Hashtable<String, MultiLabelLearner> table = new Hashtable<String, MultiLabelLearner>();
	Hashtable<String, Prediction> tablePredictions = new Hashtable<String, Prediction>();
	
	/**
	 * Constructor
	 */
	public Evaluator()
	{
		super();
	}
	

	@Override
	public Comparator<IFitness> getComparator() {
		return COMPARATOR;
	}
	
	public void setFullTrainData(MultiLabelInstances fullTrainData) {
		this.fullTrainData = fullTrainData;
	}
	
	public void setTable(Hashtable<String, MultiLabelLearner> table) {
		this.table = table;
	}
	
	public void setTablePredictions(Hashtable<String, Prediction> tablePredictions) {
		this.tablePredictions = tablePredictions;
	}
	
	@Override
	protected void evaluate(IIndividual ind) 
	{
		//Individual genotype (list)
		String gen = ((StringTreeIndividual)ind).getGenotype();
		
		Utils utils = new Utils();
		double fitness = utils.countLeaves(gen);
		
//		System.out.println("gen: " + gen);
		Prediction pred = reduce(gen);
		
		fitness = eval(pred, fullTrainData);
//		System.out.println("fitness: " + fitness);
//		System.exit(1);
		
		//Set individual fitness
		ind.setFitness(new SimpleValueFitness(fitness));

	}
	
	protected double eval(Prediction pred, MultiLabelInstances mlData) {
		int[] labelIndices = mlData.getLabelIndices();
		byte [] ground = new byte[mlData.getNumLabels()];
		
		double exF = 0.0;
		
		for(int i=0; i<mlData.getNumInstances(); i++) {
			for(int j=0; j<mlData.getNumLabels(); j++) {
				ground[j] = (byte) mlData.getDataSet().get(i).value(labelIndices[j]);
			}
			
			exF += evalInstance(pred.bip[i], ground);
			
		}
		
		return exF/mlData.getNumInstances();
	}
	
	protected double evalInstance(byte[] pred, byte[] ground) {
		double num=0, den=0;
		
		for(int i=0; i<pred.length; i++) {
			if(pred[i] > 0) {
				den++;
			}
			if(ground[i] > 0) {
				den++;
			}
			if(pred[i]>0 && ground[i]>0) {
				num += 2;
			}
		}
		
		return num/den;
	}
	

	public Prediction reduce(String ind) {
		Pattern pattern = Pattern.compile("\\((_?\\d+ )+_?\\d+\\)");
		
		Matcher m = pattern.matcher(ind);
		int count = 0;
		Prediction pred = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
		
		while(m.find()) {
			pred = combine(m.group(0), table);
			tablePredictions.put("_"+count, pred);
			ind = ind.substring(0, m.start()) + "_" + count + ind.substring(m.end(), ind.length());
			count++;
			m = pattern.matcher(ind);
		}
		
//		System.out.println(Arrays.toString(pred.bip[0]));
		return pred;
	}
	
	protected Prediction combine(String s, Hashtable<String, MultiLabelLearner> table){
		Prediction pred = new Prediction(fullTrainData.getNumInstances(), fullTrainData.getNumLabels());
		
		Pattern pattern = Pattern.compile("\\d+");
		Matcher m;
		int n;
		int nPreds = 0;
//		System.out.println("s: " + s);
		
		String [] pieces = s.split(" ");
		for(String piece : pieces) {
//			System.out.println("piece: " + piece);
			m = pattern.matcher(piece);
			m.find();
			n = Integer.parseInt(m.group(0));
//			System.out.println("n: " + n);
			if(piece.contains("_")) {
				pred.addPrediction(tablePredictions.get("_"+n));
				table.remove("_"+n);
				nPreds++;
			}
			else {
//				pred.addPrediction(getPredictions(table.get(String.valueOf(n))));
				pred.addPrediction(tablePredictions.get(String.valueOf(n)));
				nPreds++;
			}
		}
		
//		System.out.println("sum: " + Arrays.toString(pred.bip[0]));
		pred.divideAndThresholdPrediction(nPreds, 0.5);
//		System.out.println("div: " + Arrays.toString(pred.bip[0]));
		
		return pred;
	}
	
	protected byte[][] getPredictions(MultiLabelLearner learner){
		byte[][] bip = new byte[fullTrainData.getNumInstances()][fullTrainData.getNumLabels()];
		
		try {
			for(int i=0; i<fullTrainData.getNumInstances(); i++) {
				boolean[] boolPred = learner.makePrediction(fullTrainData.getDataSet().get(i)).getBipartition();
				for(int j=0; j<fullTrainData.getNumLabels(); j++) {
					if(boolPred[j]) {
						bip[i][j] = 1;
					}
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return bip;
		
	}
	
}
