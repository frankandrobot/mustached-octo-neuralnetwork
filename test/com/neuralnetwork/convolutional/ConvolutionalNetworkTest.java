package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.ActivationFunctions;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class ConvolutionalNetworkTest
{
    @Test
    public void testOutputSingleConvolutionLayer()
    {
        final DenseMatrix64F input = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

        MNeuron neuron = new MNeuron(phi, weights);
        FeatureMap.MapFunction mapFunction = new FeatureMap.ConvolutionFunction(neuron);
        mapFunction.setReceptiveFieldSize(2*2);

        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.set1DInputSize(3)
               .setMapFunction(mapFunction);
        FeatureMap featureMap = new FeatureMap(builder);

        ConvolutionalNetwork.Builder netBuilder = new ConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(featureMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        ConvolutionalNetwork network = new ConvolutionalNetwork(netBuilder);

        DenseMatrix64F output = network.output(input);

        assertThat(output.get(0,0), is(featureMap.output(input, 0, 0)));
        assertThat(output.get(1,0), is(featureMap.output(input, 1, 0)));
        assertThat(output.get(0,1), is(featureMap.output(input, 0, 1)));
        assertThat(output.get(1,1), is(featureMap.output(input, 1, 1)));

        final double o11 = input.get(0,0)*weights[0] + input.get(0,1)*weights[1]
                + input.get(1,0)*weights[2] + input.get(1,1)*weights[3]
                + weights[4];

        assertThat(network.getLayer(0).mInducedLocalField.get(0,0), is(o11));
        assertThat(network.getLayer(0).mImpulseFunction.get(0,0), is(phi.apply(o11)));

        final double o12 = input.get(0,1)*weights[0] + input.get(0,2)*weights[1]
                + input.get(1,1)*weights[2] + input.get(1,2)*weights[3]
                + weights[4];

        assertThat(network.getLayer(0).mInducedLocalField.get(0,1), is(o12));
        assertThat(network.getLayer(0).mImpulseFunction.get(0,1), is(phi.apply(o12)));

        final double o21 = input.get(1,0)*weights[0] + input.get(1,1)*weights[1]
                + input.get(2,0)*weights[2] + input.get(2,1)*weights[3]
                + weights[4];

        assertThat(network.getLayer(0).mInducedLocalField.get(1,0), is(o21));
        assertThat(network.getLayer(0).mImpulseFunction.get(1,0), is(phi.apply(o21)));
    }

    @Test
    public void testOutputSubsampleSingleLayer()
    {
        final DenseMatrix64F input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final double[] weights = {0.3, 0.4};

        final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

        MNeuron neuron = new MNeuron(phi, weights);
        FeatureMap.MapFunction mapFunction = new FeatureMap.SubSamplingFunction(neuron);
        mapFunction.setReceptiveFieldSize(2*2);

        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.set1DInputSize(4)
               .setMapFunction(mapFunction);
        FeatureMap featureMap = new FeatureMap(builder);

        ConvolutionalNetwork.Builder netBuilder = new ConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(featureMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        ConvolutionalNetwork network = new ConvolutionalNetwork(netBuilder);

        DenseMatrix64F output = network.output(input);

        assertThat(output.get(0,0), is(featureMap.output(input, 0, 0)));
        assertThat(output.get(1,0), is(featureMap.output(input, 1, 0)));
        assertThat(output.get(0,1), is(featureMap.output(input, 0, 1)));
        assertThat(output.get(1,1), is(featureMap.output(input, 1, 1)));

        final double o11 = (1 + 2 + 5 + 6) * weights[0] + weights[1];

        assertThat(network.getLayer(0).mInducedLocalField.get(0,0), is(o11));
        assertThat(network.getLayer(0).mImpulseFunction.get(0,0), is(phi.apply(o11)));

        final double o12 = (3 + 4 + 7 + 8) * weights[0] + weights[1];

        assertThat(network.getLayer(0).mInducedLocalField.get(0,1), is(o12));
        assertThat(network.getLayer(0).mImpulseFunction.get(0,1), is(phi.apply(o12)));

        final double o21 = (9 + 10 + 13 + 14) * weights[0] + weights[1];

        assertThat(network.getLayer(0).mInducedLocalField.get(1,0), is(o21));
        assertThat(network.getLayer(0).mImpulseFunction.get(1,0), is(phi.apply(o21)));
    }

    @Test
    public void testTwoLayer()
    {
        //construct convolution layer
        final DenseMatrix64F input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final double[] weights = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};

        final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

        //build first layer
        MNeuron neuron = new MNeuron(phi, weights);
        FeatureMap.MapFunction convolutionFunc = new FeatureMap.ConvolutionFunction(neuron);
        convolutionFunc.setReceptiveFieldSize(3*3);

        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.set1DInputSize(4)
               .setMapFunction(convolutionFunc);
        FeatureMap convolutionMap = new FeatureMap(builder);

        assertThat(convolutionMap.getFeatureMap().numCols, is(2));
        assertThat(convolutionMap.getFeatureMap().numRows, is(2));

        //build second layer
        final double[] weights2 = {0.3, 0.4};
        builder = new FeatureMap.Builder();
        MNeuron neuron2 = new MNeuron(phi, weights2);
        FeatureMap.MapFunction subsampFunc = new FeatureMap.SubSamplingFunction(neuron2);
        subsampFunc.setReceptiveFieldSize(2*2);

        builder.set1DInputSize(2)
               .setMapFunction(subsampFunc);
        FeatureMap subsamplingMap = new FeatureMap(builder);

        assertThat(subsamplingMap.getFeatureMap().numCols, is(1));
        assertThat(subsamplingMap.getFeatureMap().numRows, is(1));

        //build network
        ConvolutionalNetwork.Builder netBuilder = new ConvolutionalNetwork.Builder();
        netBuilder.setGlobalActivationFunction(phi)
                  .setLayers(convolutionMap, subsamplingMap)
                  .setLearningParam(0.0)
                  .setMomentumParam(0.0);

        ConvolutionalNetwork network = new ConvolutionalNetwork(netBuilder);

        final DenseMatrix64F output = network.output(input);

        assertThat(output.numRows, is(1));
        assertThat(output.numCols, is(1));

        final double o11 = phi.apply(
                1*0.01 + 2*0.02 + 3*0.03
                        + 5*0.04 + 6*0.05 + 7*0.06
                        + 9*0.07 + 10*0.08 + 11*0.09 + 0.1);
        final double o12 = phi.apply(
                2*0.01 + 3*0.02 + 4*0.03
                        + 6*0.04 + 7*0.05 + 8*0.06
                        + 10*0.07 + 11*0.08 + 12*0.09 + 0.1);
        final double o21 = phi.apply(
                5*0.01 + 6*0.02 + 7*0.03
                        + 9*0.04 + 10*0.05 + 11*0.06
                        + 13*0.07 + 14*0.08 + 15*0.09 + 0.1);
        final double o22 = phi.apply(
                6*0.01 + 7*0.02 + 8*0.03
                        + 10*0.04 + 11*0.05 + 12*0.06
                        + 14*0.07 + 15*0.08 + 16*0.09 + 0.1);

        final double r11 = phi.apply( (o11 + o12 + o21 + o22)*weights2[0] + weights2[1] );
        assertThat(r11, is(output.get(0,0)));
    }
}
