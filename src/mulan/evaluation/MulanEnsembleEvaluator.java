package mulan.evaluation;

import java.util.List;

import g3pkemlc.EMLC;
import mulan.classifier.MultiLabelLearner;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;

public class MulanEnsembleEvaluator extends Evaluator {

	int seed;
	
	@Override
	public void setSeed(int seed) {
		this.seed = seed;
	}
	
	public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances data, List<Measure> measures) throws IllegalArgumentException, Exception {
		((EMLC)learner).resetSeed(seed);
		return super.evaluate(learner, data, measures);
	}
	
}
