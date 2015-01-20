package com.neuralnetwork.cnn;

import com.neuralnetwork.cnn.filter.SimpleConvolutionFilter;
import com.neuralnetwork.cnn.filter.SimpleSamplingFilter;
import com.neuralnetwork.cnn.map.ConvolutionMap;
import com.neuralnetwork.cnn.map.ConvolutionMapBuilder;
import com.neuralnetwork.cnn.map.SamplingMap;
import com.neuralnetwork.cnn.map.SamplingMapBuilder;
import com.neuralnetwork.core.ActivationFunctions;
import com.neuralnetwork.core.neuron.Neuron;
import com.neuralnetwork.nn.layer.NNLayer;
import com.neuralnetwork.nn.layer.NNLayerBuilder;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import static org.hamcrest.core.Is.is;
import static org.junit.Assert.assertThat;

public class CnnTest
{

    private final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    @Test
    public void testTwoLayer() throws Exception
    {
        //construct convolution layer
        final DenseMatrix64F input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final double[] weights = {0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};

        //build first layer
        ConvolutionMap convolutionMap = new ConvolutionMapBuilder()
                .setNeuron(new Neuron(phi, weights))
                .setFilter(new SimpleConvolutionFilter())
                .set1DInputSize(4)
                .build();


        final double[] weights2 = {0.4, 0.3, 0.3, 0.3, 0.3};

        //build second layer
        SamplingMap samplingMap = new SamplingMapBuilder()
                .setNeuron(new Neuron(phi, weights2))
                .setFilter(new SimpleSamplingFilter())
                .set1DInputSize(2)
                .build();

        //build network
        CNN net = new CNNBuilder()
                .setLayer(new CNNConnection(convolutionMap))
                .setLayer(new CNNConnection(samplingMap, convolutionMap))
                .build();


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

        final double r11 = phi.apply( (o11 + o12 + o21 + o22)*weights2[1] + weights2[0] );

        final DenseMatrix64F output = net.generateOutput(input);

        assertThat(1, is(output.numRows));
        assertThat(1, is(output.numCols));
        assertThat(output.get(0,0), is(r11));
    }

    @Test
    public void testMultiKernel() throws Exception
    {
        //construct convolution layer
        final DenseMatrix64F input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final double[] weights = {0.1, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09};

        //build first layer
        ConvolutionMap convolutionMap = new ConvolutionMapBuilder()
                .setNeuron(new Neuron(phi,weights), new Neuron(phi,weights))
                .setFilter(new SimpleConvolutionFilter(), new SimpleConvolutionFilter())
                .set1DInputSize(4)
                .build();


        //build network
        CNN net = new CNNBuilder()
                .setLayer(new CNNConnection(convolutionMap))
                .build();


        final double o11 = phi.apply(
                2.0*(1*0.01 + 2*0.02 + 3*0.03
                        + 5*0.04 + 6*0.05 + 7*0.06
                        + 9*0.07 + 10*0.08 + 11*0.09) + 0.1);
        final double o12 = phi.apply(
                2.0*(2*0.01 + 3*0.02 + 4*0.03
                        + 6*0.04 + 7*0.05 + 8*0.06
                        + 10*0.07 + 11*0.08 + 12*0.09) + 0.1);
        final double o21 = phi.apply(
                2.0*(5*0.01 + 6*0.02 + 7*0.03
                        + 9*0.04 + 10*0.05 + 11*0.06
                        + 13*0.07 + 14*0.08 + 15*0.09) + 0.1);
        final double o22 = phi.apply(
                2.0*(6*0.01 + 7*0.02 + 8*0.03
                        + 10*0.04 + 11*0.05 + 12*0.06
                        + 14*0.07 + 15*0.08 + 16*0.09) + 0.1);

        final DenseMatrix64F output = net.generateOutput(input, input);

        assertThat(output.get(0,0), is(o11));
        assertThat(output.get(0,1), is(o12));
        assertThat(output.get(1,0), is(o21));
        assertThat(output.get(1,1), is(o22));
    }

    @Test
    public void testFullyConnectedLayer() throws Exception
    {
        DenseMatrix64F input = new DenseMatrix64F(2,2,true,new double[] {
           1, 2,
           3, 4
        });


        double[] weights = { -0.5, 0.1, 0.1, 0.1, 0.1 };

        SamplingMap samplingMap1 = new SamplingMapBuilder()
                .setFilter(new SimpleSamplingFilter())
                .set1DInputSize(2)
                .setNeuron(new Neuron(phi, weights))
                .build();

        SamplingMap samplingMap2 = new SamplingMapBuilder()
                .setFilter(new SimpleSamplingFilter())
                .set1DInputSize(2)
                .setNeuron(new Neuron(phi, weights))
                .build();


        double[] weights2 = { -0.4, 0.1, 0.2 };

        NNLayer nnLayer = new NNLayerBuilder()
                .setNeuron(phi, weights2)
                .build();


        CNN net = new CNNBuilder()
                .setLayer(new CNNConnection(samplingMap1), new CNNConnection(samplingMap2))
                .setLayer(new CNNConnection(nnLayer, samplingMap1, samplingMap2))
                .build();


        DenseMatrix64F output1 = samplingMap1.generateOutput(new DenseMatrix64F[] { input });
        DenseMatrix64F output2 = samplingMap2.generateOutput(new DenseMatrix64F[] { input });

        double output = phi.apply(
                output1.get(0,0)*0.1 + output2.get(0,0)*0.2 - 0.4
        );

        DenseMatrix64F actual = net.generateOutput(new DenseMatrix64F[] { input });

        assertThat(actual.get(0,0), is(output));
    }
}