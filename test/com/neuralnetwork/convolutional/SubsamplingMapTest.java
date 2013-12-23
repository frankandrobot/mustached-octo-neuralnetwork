package com.neuralnetwork.convolutional;

import com.neuralnetwork.core.ActivationFunctions;
import org.ejml.data.DenseMatrix64F;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class SubsamplingMapTest
{

    private final ActivationFunctions.SigmoidUnityFunction phi = new ActivationFunctions.SigmoidUnityFunction();

    @Test
    public void testConvolution()
    {
        final DenseMatrix64F input = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        FeatureMap.Builder builder = new FeatureMap.Builder();

        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);
        FeatureMap featureMap = new ConvolutionMap(builder);

        featureMap.output(input);

        //1 2 3 4 5 ... 24 25 26 27 28
        assertThat(featureMap.mFeatureMap.numRows, is(2) );
        assertThat(featureMap.mFeatureMap.numCols, is(2) );

        double o11 = input.get(0,0)*weights[0] + input.get(0,1)*weights[1]
                + input.get(1,0)*weights[2] + input.get(1,1)*weights[3]
                + weights[4];
        o11 = phi.apply(o11);

        assertThat(featureMap.mFeatureMap.get(0, 0), is(o11));

        double o12 = input.get(0,1)*weights[0] + input.get(0,2)*weights[1]
                + input.get(1,1)*weights[2] + input.get(1,2)*weights[3]
                + weights[4];
        o12 = phi.apply(o12);

        assertThat(featureMap.mFeatureMap.get(0,1), is(o12));

        double o21 = input.get(1,0)*weights[0] + input.get(1,1)*weights[1]
                + input.get(2,0)*weights[2] + input.get(2,1)*weights[3]
                + weights[4];
        o21 = phi.apply(o21);

        assertThat(featureMap.mFeatureMap.get(1,0), is(o21));
    }

    @Test
    public void testSubsampling()
    {
        final DenseMatrix64F input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final double[] weights = {0.3, 0.4};

        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(4);

        FeatureMap featureMap = new FeatureMap.SubSamplingMap(builder);

        featureMap.output(input);

        //1 2 3 4 5 ... 24 25 26 27 28
        assertThat(featureMap.mFeatureMap.numCols, is(2) );
        assertThat(featureMap.mFeatureMap.numRows, is(2) );

        double o11 = (1 + 2 + 5 + 6);
        o11 = o11 * weights[0] + weights[1];
        o11 = phi.apply(o11);

        assertThat(featureMap.mFeatureMap.get(0,0), is(o11));

        double o12 = (3 + 4 + 7 + 8);
        o12 = o12 * weights[0] + weights[1];
        o12 = phi.apply(o12);

        assertThat(featureMap.mFeatureMap.get(0,1), is(o12));

        double o21 = (9 + 10 + 13 + 14);
        o21 = o21 * weights[0] + weights[1];
        o21 = phi.apply(o21);

        assertThat(featureMap.mFeatureMap.get(1,0), is(o21));

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


        //build first layer
        final double[] weights = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1};
        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.set1DInputSize(4);
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(3*3);

        FeatureMap convolutionMap = new ConvolutionMap(builder);

        final DenseMatrix64F output = convolutionMap.output(input);
        assertThat(output.numCols, is(2));
        assertThat(output.numRows, is(2));

        //build second layer
        final double[] weights2 = {0.3, 0.4};
        builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights2));
        builder.setReceptiveFieldSize(2*2);
        builder.set1DInputSize(2);

        FeatureMap subsamplingMap = new FeatureMap.SubSamplingMap(builder);

        final DenseMatrix64F rslt = subsamplingMap.output(output);
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

    @Test
    public void testConvolutionRawOutputFuncs()
    {
        final DenseMatrix64F input = new DenseMatrix64F(3,3,true, new double[] {
                1, 2, 3
                ,4, 5, 6
                ,7, 8, 9
        });

        final double[] weights = {0.1, 0.2, 0.3, 0.4, 0.5};

        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(3);

        FeatureMap featureMap = new ConvolutionMap(builder);

        DenseMatrix64F output = featureMap.calculateFeatureMap(input);

        assertThat(output.get(0,0), is(featureMap.output(input, 0, 0)));
        assertThat(output.get(1,0), is(featureMap.output(input, 1, 0)));
        assertThat(output.get(0,1), is(featureMap.output(input, 0, 1)));
        assertThat(output.get(1,1), is(featureMap.output(input, 1, 1)));

        final double o11 = input.get(0,0)*weights[0] + input.get(0,1)*weights[1]
                + input.get(1,0)*weights[2] + input.get(1,1)*weights[3]
                + weights[4];

        assertThat(featureMap.rawoutput(input,0,0), is(o11));

        final double o12 = input.get(0,1)*weights[0] + input.get(0,2)*weights[1]
                + input.get(1,1)*weights[2] + input.get(1,2)*weights[3]
                + weights[4];

        assertThat(featureMap.rawoutput(input, 0, 1), is(o12));

        final double o21 = input.get(1,0)*weights[0] + input.get(1,1)*weights[1]
                + input.get(2,0)*weights[2] + input.get(2,1)*weights[3]
                + weights[4];

        assertThat(featureMap.rawoutput(input,1,0), is(o21));
    }

    @Test
    public void testSubsamplingRawOutputFuncs()
    {
        final DenseMatrix64F input = new DenseMatrix64F(4,4,true, new double[] {
                1, 2, 3, 4
                ,5, 6, 7, 8
                ,9, 10, 11, 12
                ,13, 14, 15, 16
        });

        final double[] weights = {0.3, 0.4};

        FeatureMap.Builder builder = new FeatureMap.Builder();
        builder.setNeuron(new MNeuron(phi, weights));
        builder.setReceptiveFieldSize(2 * 2);
        builder.set1DInputSize(4);

        FeatureMap featureMap = new FeatureMap.SubSamplingMap(builder);

        DenseMatrix64F output = featureMap.calculateFeatureMap(input);

        assertThat(output.get(0,0), is(featureMap.output(input, 0, 0)));
        assertThat(output.get(1,0), is(featureMap.output(input, 1, 0)));
        assertThat(output.get(0,1), is(featureMap.output(input, 0, 1)));
        assertThat(output.get(1,1), is(featureMap.output(input, 1, 1)));

        final double o11 = (1 + 2 + 5 + 6) * weights[0] + weights[1];

        assertThat(featureMap.rawoutput(input, 0, 0), is(o11));

        final double o12 = (3 + 4 + 7 + 8) * weights[0] + weights[1];

        assertThat(featureMap.rawoutput(input,0,1), is(o12));

        final double o21 = (9 + 10 + 13 + 14) * weights[0] + weights[1];

        assertThat(featureMap.rawoutput(input,1,0), is(o21));
    }
}