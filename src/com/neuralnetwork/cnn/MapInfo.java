package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.ICnnMap;
import org.ejml.data.DenseMatrix64F;

import java.util.LinkedList;
import java.util.List;

class MapInfo
{
    /**
     * could be null
     */
    ICnnMap map;

    DenseMatrix64F yOutput;

    private List<MapInfo> lInputMaps = new LinkedList<MapInfo>();
    private DenseMatrix64F[] aInputs;


    public MapInfo(ICnnMap map)
    {
        if (map != null)
        {
            this.map = map;
            yOutput = new DenseMatrix64F(map.getOutputDim().rows, map.getOutputDim().cols);
        }
    }

    public MapInfo(DenseMatrix64F input)
    {
        yOutput = input;
    }

    public void addInput(MapInfo inputMap)
    {
        lInputMaps.add(inputMap);

        aInputs = new DenseMatrix64F[lInputMaps.size()];
    }

    /**
     * Valid only if has map
     */
    public void generateOutput()
    {
        if (aInputs != null)
        {
            this.yOutput = map.generateOutput(getInputs());
        }
    }

    public List<MapInfo> getInputMapInfo()
    {
        return lInputMaps;
    }

    private List<ICnnMap> getInputMaps()
    {
        List<ICnnMap> maps = new LinkedList<ICnnMap>();

        for(MapInfo mapInfo:lInputMaps)
        {
            maps.add(mapInfo.map);
        }

        return maps;
    }

    private DenseMatrix64F[] getInputs()
    {
        int len = 0;

        for(MapInfo map:lInputMaps)
        {
            aInputs[len++] = map.yOutput;
        }

        return aInputs;
    }

    public void validateInputs()
    {
        map.validateInputs(getInputMaps());
    }
}
