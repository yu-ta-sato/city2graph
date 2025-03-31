import pytest
import json
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from city2graph.utils import (
    has_tunnel,
    extract_tunnel_segments,
    line_substring,
    get_non_tunnel_parts,
    process_segment
)

class TestHasTunnel:
    def test_has_tunnel_valid_json(self):
        # Valid JSON with tunnel flag
        assert has_tunnel('[{"values": {"is_tunnel": true}}]') is True
        
        # Valid JSON without tunnel flag
        assert has_tunnel('[{"values": {"some_other_flag": true}}]') is False
        
        # Empty JSON array
        assert has_tunnel('[]') is False
    
    def test_has_tunnel_invalid_input(self):
        # None input
        assert has_tunnel(None) is False
        
        # Non-string input
        assert has_tunnel(123) is False
        
        # Invalid JSON format
        assert has_tunnel('{invalid json}') is False
        
        # String but not JSON
        assert has_tunnel('not a json') is False

    def test_has_tunnel_complex_json(self, tunnel_road_flags):
        # Test with fixture data
        assert has_tunnel(tunnel_road_flags["full"]) is True
        assert has_tunnel(tunnel_road_flags["partial"]) is True
        assert has_tunnel(tunnel_road_flags["multiple"]) is True
        assert has_tunnel(tunnel_road_flags["none"]) is False


class TestExtractTunnelSegments:
    def test_extract_tunnel_segments_complete(self):
        # Full tunnel (implied by missing 'between')
        road_flags = '[{"values": {"is_tunnel": true}}]'
        assert extract_tunnel_segments(road_flags) == [(0, 1)]
    
    def test_extract_tunnel_segments_partial(self):
        # Partial tunnel segment
        road_flags = '[{"values": {"is_tunnel": true}, "between": [0.2, 0.7]}]'
        assert extract_tunnel_segments(road_flags) == [(0.2, 0.7)]
        
        # Multiple tunnel segments
        road_flags = '''[
            {"values": {"is_tunnel": true}, "between": [0.2, 0.4]},
            {"values": {"is_tunnel": true}, "between": [0.6, 0.9]}
        ]'''
        assert extract_tunnel_segments(road_flags) == [(0.2, 0.4), (0.6, 0.9)]
    
    def test_extract_tunnel_segments_invalid_input(self):
        # None input
        assert extract_tunnel_segments(None) == []
        
        # Non-string input
        assert extract_tunnel_segments(123) == []
        
        # Invalid JSON
        assert extract_tunnel_segments('{invalid}') == []
        
        # Valid JSON but invalid between values
        road_flags = '[{"values": {"is_tunnel": true}, "between": "invalid"}]'
        assert extract_tunnel_segments(road_flags) == [(0, 1)]
    
    def test_extract_tunnel_segments_fixture(self, tunnel_road_flags):
        assert extract_tunnel_segments(tunnel_road_flags["full"]) == [(0, 1)]
        assert extract_tunnel_segments(tunnel_road_flags["partial"]) == [(0.3, 0.7)]
        assert extract_tunnel_segments(tunnel_road_flags["multiple"]) == [(0.2, 0.3), (0.6, 0.8)]
        assert extract_tunnel_segments(tunnel_road_flags["none"]) == []


class TestLineSubstring:
    def test_line_substring_full(self, simple_line):
        # Extract full line
        result = line_substring(simple_line, 0, 1)
        assert result.equals(simple_line)
    
    def test_line_substring_middle(self, simple_line):
        # Extract middle section (2,0) to (7,0)
        result = line_substring(simple_line, 0.2, 0.7)
        expected = LineString([(2, 0), (7, 0)])
        assert result.equals(expected)
    
    def test_line_substring_start(self, simple_line):
        # Extract start section (0,0) to (5,0)
        result = line_substring(simple_line, 0, 0.5)
        expected = LineString([(0, 0), (5, 0)])
        assert result.equals(expected)
    
    def test_line_substring_end(self, simple_line):
        # Extract end section (5,0) to (10,0)
        result = line_substring(simple_line, 0.5, 1)
        expected = LineString([(5, 0), (10, 0)])
        assert result.equals(expected)
    
    def test_line_substring_complex(self, complex_line):
        # Test with complex line
        result = line_substring(complex_line, 0.3, 0.7)
        # Validate that result is a LineString
        assert isinstance(result, LineString)
        # Validate that length is roughly 40% of the original
        assert abs(result.length / complex_line.length - 0.4) < 0.01
    
    def test_line_substring_invalid_line(self):
        # Test with non-LineString geometry
        point = Point(0, 0)
        result = line_substring(point, 0, 1)
        assert result is None
    
    def test_line_substring_invalid_parameters(self, simple_line):
        # Test with invalid percentage parameters
        assert line_substring(simple_line, -0.1, 0.5) is None
        assert line_substring(simple_line, 0.5, 1.1) is None
        assert line_substring(simple_line, 0.7, 0.5) is None
        assert line_substring(simple_line, 0, 0) is None


class TestGetNonTunnelParts:
    def test_get_non_tunnel_parts_no_tunnels(self, simple_line):
        # No tunnel segments
        result = get_non_tunnel_parts(simple_line, [])
        assert result.equals(simple_line)
    
    def test_get_non_tunnel_parts_full_tunnel(self, simple_line):
        # Full tunnel
        result = get_non_tunnel_parts(simple_line, [(0, 1)])
        assert result is None
    
    def test_get_non_tunnel_parts_middle_tunnel(self, simple_line):
        # Middle tunnel
        result = get_non_tunnel_parts(simple_line, [(0.4, 0.6)])
        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 2
        # First part is 0-0.4 of the original line
        assert abs(result.geoms[0].length / simple_line.length - 0.4) < 0.01
        # Second part is 0.6-1.0 of the original line
        assert abs(result.geoms[1].length / simple_line.length - 0.4) < 0.01
    
    def test_get_non_tunnel_parts_start_tunnel(self, simple_line):
        # Tunnel at start
        result = get_non_tunnel_parts(simple_line, [(0, 0.3)])
        assert isinstance(result, LineString)
        # Result is 0.3-1.0 of the original line
        assert abs(result.length / simple_line.length - 0.7) < 0.01
    
    def test_get_non_tunnel_parts_end_tunnel(self, simple_line):
        # Tunnel at end
        result = get_non_tunnel_parts(simple_line, [(0.7, 1)])
        assert isinstance(result, LineString)
        # Result is 0-0.7 of the original line
        assert abs(result.length / simple_line.length - 0.7) < 0.01
    
    def test_get_non_tunnel_parts_multiple_tunnels(self, simple_line):
        # Multiple tunnels
        result = get_non_tunnel_parts(simple_line, [(0.2, 0.3), (0.6, 0.8)])
        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 3
        # Parts should be 0-0.2, 0.3-0.6, 0.8-1.0 of the original line
        total_fraction = 0.2 + 0.3 + 0.2  # Sum of non-tunnel fractions
        assert abs(result.length / simple_line.length - total_fraction) < 0.01
    
    def test_get_non_tunnel_parts_overlapping_tunnels(self, simple_line):
        # Overlapping tunnels should be handled correctly
        result = get_non_tunnel_parts(simple_line, [(0.2, 0.5), (0.4, 0.7)])
        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 2
        # Parts should be 0-0.2 and 0.7-1.0 of the original line
        total_fraction = 0.2 + 0.3  # Sum of non-tunnel fractions
        assert abs(result.length / simple_line.length - total_fraction) < 0.01


class TestProcessSegment:
    def test_process_segment_non_road(self, simple_line):
        # Test with non-road segment
        row = type('obj', (object,), {
            'geometry': simple_line,
            'subtype': 'footway',
            'road_flags': None
        })
        result = process_segment(row)
        assert result.equals(simple_line)
    
    def test_process_segment_road_no_tunnel(self, simple_line):
        # Test with road segment, no tunnel
        row = type('obj', (object,), {
            'geometry': simple_line,
            'subtype': 'road',
            'road_flags': '[]'
        })
        result = process_segment(row)
        assert result.equals(simple_line)
    
    def test_process_segment_road_with_tunnel(self, simple_line, tunnel_road_flags):
        # Test with road segment, middle tunnel
        row = type('obj', (object,), {
            'geometry': simple_line,
            'subtype': 'road',
            'road_flags': tunnel_road_flags["partial"]
        })
        result = process_segment(row)
        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 2
    
    def test_process_segment_road_full_tunnel(self, simple_line, tunnel_road_flags):
        # Test with road segment, full tunnel
        row = type('obj', (object,), {
            'geometry': simple_line,
            'subtype': 'road',
            'road_flags': tunnel_road_flags["full"]
        })
        result = process_segment(row)
        assert result is None
    
    def test_process_segment_invalid_geometry(self):
        # Test with invalid geometry
        row = type('obj', (object,), {
            'geometry': Point(0, 0),
            'subtype': 'road',
            'road_flags': '[{"values": {"is_tunnel": true}}]'
        })
        result = process_segment(row)
        assert result.equals(Point(0, 0))
    
    def test_process_segment_with_get_method(self, simple_line, tunnel_road_flags):
        # Test with dict-like object using get method
        class DictLike:
            def __init__(self, data):
                self.data = data
            def get(self, key, default=None):
                return self.data.get(key, default)
            def __getattr__(self, key):
                return self.data.get(key)
                
        row = DictLike({
            'geometry': simple_line,
            'subtype': 'road',
            'road_flags': tunnel_road_flags["partial"],
            'id': 'test_id'
        })
        
        result = process_segment(row)
        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 2
