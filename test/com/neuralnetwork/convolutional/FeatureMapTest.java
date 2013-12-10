package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.Neuron;
import org.junit.Test;

public class FeatureMapTest
{
    @Test
    public void testConvolution()
    {
        final double[][] input = {
                new double[]{1, 2, 3}
                ,new double[]{4, 5, 6}
                ,new double[]{7, 8, 9}
        };

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        FeatureMap.Builder builder = new FeatureMap.Builder();
        Neuron neuron = new Neuron(new ActivationFunctions.SigmoidUnityFunction(), weights);
        builder.setSharedNeuron(neuron);
        builder.setInputSize(3);
        builder.setReceptiveFieldSize(2);
        FeatureMap featureMap = new FeatureMap(builder);

        featureMap.output(input);

    }
}
