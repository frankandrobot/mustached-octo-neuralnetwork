package com.neuralnetwork.convolutional.convolutionmap;

import com.neuralnetwork.convolutional.MNeuron;
import com.neuralnetwork.core.ActivationFunctions;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class FeatureMapTest
{

    private final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

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


        //build first layer
        final double[] weights = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};
        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.set1DInputSize(4);
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(3*3);

        FeatureMap convolutionMap = new ConvolutionMapLayerOld(builder);

        final DenseMatrix64F output = convolutionMap.constructOutput(input);
        assertThat(output.numCols, is(2));
        assertThat(output.numRows, is(2));

        //build second layer
        final double[] weights2 = {0.3, 0.4};
        builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights2));
        builder.setReceptiveFieldSize(2*2);
        builder.set1DInputSize(2);

        FeatureMap subsamplingMap = new SubSamplingMapOld(builder);

        final DenseMatrix64F rslt = subsamplingMap.constructOutput(output);
        assertThat(rslt.numCols, is(1));
        assertThat(rslt.numRows, is(1));

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

        assertThat(o11, is(output.get(0,0)));
        assertThat(o12, is(output.get(0,1)));
        assertThat(o21, is(output.get(1,0)));
        assertThat(o22, is(output.get(1,1)));

        final double r11 = phi.apply( (o11 + o12 + o21 + o22)*weights2[0] + weights2[1] );
        assertThat(r11, is(rslt.get(0,0)));
    }
}
