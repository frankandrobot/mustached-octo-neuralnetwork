package com.neuralnetwork.nn.backprop;

import com.neuralnetwork.core.Example;
import com.neuralnetwork.core.interfaces.INeuralLayer;
import org.ejml.data.DenseMatrix64F;

public class NNBackprop
{

    public void go(double eta, INeuralLayer[] layers, Example... examples)
    {
        NNBackpropHelper backprop = new NNBackpropHelper(layers);

        for(Example example:examples)
        {
            backprop.go(example);

        }

        //update weights
        DenseMatrix64F[] aLearningTerms = backprop.getCumulativeLearningTermsMinusEta();

        for(int i=0; i<layers.length; i++)
        {
            DenseMatrix64F weights = layers[i].getWeightMatrix();
            DenseMatrix64F learningTerms = aLearningTerms[i];

            for(int k=0; k<weights.data.length; k++)
            {
                weights.data[k] += eta * learningTerms.data[k];
            }
        }


    }
}
