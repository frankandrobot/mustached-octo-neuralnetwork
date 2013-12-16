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
        FeatureMap.MapFunction mapFunction = new FeatureMap.ConvolutionFunction(neuron);
        mapFunction.setReceptiveFieldSize(2*2);

        builder.setInputSize(3)
               .setMapFunction(mapFunction);
        FeatureMap featureMap = new FeatureMap(builder);

        featureMap.output(input);

        //1 2 3 4 5 ... 24 25 26 27 28
        assertThat(featureMap.aFeatureMap.length, is(2) );

        double o11 = input[0][0]*weights[0] + input[0][1]*weights[1]
                + input[1][0]*weights[2] + input[1][1]*weights[3]
                + weights[4];
        o11 = phi.apply(o11);

        assertThat(featureMap.aFeatureMap[0][0], is(o11));

        double o12 = input[0][1]*weights[0] + input[0][2]*weights[1]
                + input[1][1]*weights[2] + input[1][2]*weights[3]
                + weights[4];
        o12 = phi.apply(o12);

        assertThat(featureMap.aFeatureMap[0][1], is(o12));

        double o21 = input[1][0]*weights[0] + input[1][1]*weights[1]
                + input[2][0]*weights[2] + input[2][1]*weights[3]
                + weights[4];
        o21 = phi.apply(o21);

        assertThat(featureMap.aFeatureMap[1][0], is(o21));
    }

    @Test
    public void testSubsampling()
    {
        final double[][] input = {
                new double[]{1, 2, 3, 4}
                ,new double[]{5, 6, 7, 8}
                ,new double[]{9, 10, 11, 12}
                ,new double[]{13, 14, 15, 16}
        };

        final double[] weights = {0.3, 0.4};

        FeatureMap.Builder builder = new FeatureMap.Builder();
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();
        Neuron neuron = new Neuron(phi, weights);
        FeatureMap.MapFunction mapFunction = new FeatureMap.SubSamplingFunction(neuron);
        mapFunction.setReceptiveFieldSize(2*2);

        builder.setInputSize(4)
                .setMapFunction(mapFunction);
        FeatureMap featureMap = new FeatureMap(builder);

        featureMap.output(input);

        //1 2 3 4 5 ... 24 25 26 27 28
        assertThat(featureMap.aFeatureMap.length, is(2) );

        double o11 = (1 + 2 + 5 + 6);
        o11 = o11 * weights[0] + weights[1];
        o11 = phi.apply(o11);

        assertThat(featureMap.aFeatureMap[0][0], is(o11));

        double o12 = (3 + 4 + 7 + 8);
        o12 = o12 * weights[0] + weights[1];
        o12 = phi.apply(o12);

        assertThat(featureMap.aFeatureMap[0][1], is(o12));

        double o21 = (9 + 10 + 13 + 14);
        o21 = o21 * weights[0] + weights[1];
        o21 = phi.apply(o21);

        assertThat(featureMap.aFeatureMap[1][0], is(o21));

    }

    @Test
    public void testTwoLayer()
    {
        //construct convolution layer
        final double[][] input = {
                new double[]{1, 2, 3, 4}
                ,new double[]{5, 6, 7, 8}
                ,new double[]{9, 10, 11, 12}
                ,new double[]{13, 14, 15, 16}
        };

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

        FeatureMap.Builder builder = new FeatureMap.Builder();
        ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();
        Neuron neuron = new Neuron(phi, weights);
        FeatureMap.MapFunction convolutionFunc = new FeatureMap.ConvolutionFunction(neuron);
        convolutionFunc.setReceptiveFieldSize(3*3);

        builder.setInputSize(4)
               .setMapFunction(convolutionFunc);
        FeatureMap convolutionMap = new FeatureMap(builder);

        convolutionMap.output(input);

        final double[] weights2 = {0.3, 0.4};

        builder = new FeatureMap.Builder();
        Neuron neuron2 = new Neuron(phi, weights);
        FeatureMap.MapFunction subsampFunc = new FeatureMap.SubSamplingFunction(neuron2);
        subsampFunc.setReceptiveFieldSize(2*2);

        builder.setInputSize(4)
               .setMapFunction(subsampFunc);
        FeatureMap featureMap = new FeatureMap(builder);

        featureMap.output(input);
    }
}
