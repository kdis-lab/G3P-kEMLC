package mulan.evaluation;

import java.util.List;

import coeaglet.algorithm.Ensemble;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;

public class MulanEnsembleEvaluator extends Evaluator {

	public Evaluation evaluate(MultiLabelLearner learner, MultiLabelInstances data, List<Measure> measures) throws IllegalArgumentException, Exception {
		((Ensemble)learner).resetSeed();
		return super.evaluate(learner, data, measures);
	}
	
}
