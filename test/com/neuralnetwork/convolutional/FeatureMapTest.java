package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.Neuron;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

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
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();
        Neuron neuron = new Neuron(phi, weights);
        builder.setSharedNeuron(neuron);
        builder.setInputSize(3);
        builder.setReceptiveFieldSize(2);
        FeatureMap featureMap = new FeatureMap(builder);

        featureMap.output(input);

        double o11 = input[0][0]*weights[0] + input[0][1]*weights[1] + input[1][0]*weights[2] + input[1][1]*weights[3] + weights[4];
        o11 = phi.apply(o11);

        assertThat(featureMap.aFeatureMap[0][0], is(o11));
    }
}
