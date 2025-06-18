#!/usr/bin/env bash
set -e

# 1. Homebrew
if ! command -v brew >/dev/null 2>&1; then
  echo "Installing Homebrew…"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# 2. Node.js
if ! command -v node >/dev/null 2>&1; then
  echo "Installing Node.js…"
  brew install node
fi

# 3. http-server
if ! command -v http-server >/dev/null 2>&1; then
  echo "Installing http-server…"
  npm install -g http-server
fi

# 4. Scaffold project files
echo "Creating project files…"
cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SWOT Timeline</title>
  <style>
    body { font-family: sans-serif; margin:0; }
    .axis text { fill: #888; font-size: 12px; }
    .axis .active { fill: #000; font-weight: bold; }
    .quadrant text.value { font-size: 24px; fill: #333; text-anchor: middle; dominant-baseline: middle; }
    .year-display { font-size: 48px; text-anchor: middle; fill: #555; }
  </style>
</head>
<body>
  <svg id="swot-viz"></svg>
  <script src="https://unpkg.com/d3@7"></script>
  <script src="script.js"></script>
</body>
</html>
EOF

cat > script.js << 'EOF'
// data
const data = [
  { year: 2012, S:5, W:7, O:5, T:7 },
  { year: 2013, S:4, W:8, O:5, T:8 },
  { year: 2014, S:6, W:7, O:7, T:6 },
  { year: 2015, S:5, W:7, O:6, T:6 },
  { year: 2016, S:5, W:6, O:6, T:5 },
  { year: 2017, S:6, W:5, O:7, T:5 },
  { year: 2018, S:8, W:4, O:8, T:5 },
  { year: 2019, S:9, W:3, O:7, T:5 },
  { year: 2020, S:9, W:3, O:8, T:4 },
  { year: 2021, S:10, W:2, O:8, T:6 },
  { year: 2022, S:8, W:4, O:7, T:7 },
  { year: 2023, S:7, W:6, O:6, T:8 },
  { year: 2024, S:7, W:6, O:5, T:8 }
];

const margin = { top: 60, right: 40, bottom: 100, left: 40 },
      width  = 800 - margin.left - margin.right,
      height = 600 - margin.top - margin.bottom;

const svg = d3.select('#swot-viz')
  .attr('width', width + margin.left + margin.right)
  .attr('height', height + margin.top + margin.bottom);

const g = svg.append('g')
  .attr('transform', `translate(${margin.left},${margin.top})`);

// quadrant sizing
const quadW = width / 2,
      quadH = (height - 80) / 2; // leave 80px for timeline

const quadrants = [
  { name: 'Strength',    key: 'S', x: 0,      y: 0      },
  { name: 'Weakness',    key: 'W', x: quadW,  y: 0      },
  { name: 'Opportunity', key: 'O', x: 0,      y: quadH  },
  { name: 'Threat',      key: 'T', x: quadW,  y: quadH  }
];

// color scale for intensity
const colorScale = d3.scaleLinear()
  .domain([1,10])
  .range([0.2,1]); // we'll feed this into an interpolator

// draw quadrant groups
const qg = g.selectAll('.quadrant')
  .data(quadrants)
  .enter().append('g')
    .attr('class','quadrant')
    .attr('transform', d => `translate(${d.x},${d.y})`);

// background rect + label + value placeholder
qg.append('rect')
    .attr('width', quadW - 2)
    .attr('height', quadH - 2)
    .attr('stroke','#aaa')
    .attr('fill','#eee');

qg.append('text')
    .attr('x', (quadW-2)/2)
    .attr('y', 20)
    .attr('text-anchor','middle')
    .text(d => d.name);

qg.append('text')
    .attr('class','value')
    .attr('x', (quadW-2)/2)
    .attr('y', quadH/2)
    .text('');


// year big display
const yearText = g.append('text')
    .attr('class','year-display')
    .attr('x', width/2)
    .attr('y', -20)
    .text('');

// timeline
const years = data.map(d => d.year);
const x = d3.scalePoint()
    .domain(years)
    .range([0, width])
    .padding(0.5);

const timeline = g.append('g')
    .attr('class','axis')
    .attr('transform', `translate(0,${height - 80 + 30})`);

timeline.selectAll('text')
  .data(years)
  .enter().append('text')
    .attr('x', d => x(d))
    .attr('y', 0)
    .attr('text-anchor','middle')
    .text(d => d)
    .attr('fill','#888');

timeline.selectAll('line')
  .data(years)
  .enter().append('line')
    .attr('x1', d => x(d))
    .attr('x2', d => x(d))
    .attr('y1', -5)
    .attr('y2', 5)
    .attr('stroke','#aaa');

// moving dot
const dot = timeline.append('circle')
    .attr('r', 6)
    .attr('fill','red')
    .attr('cy', -10)
    .attr('cx', x(years[0]));

let idx = 0;
function update(idx){
  const d = data[idx];
  // year text
  yearText.text(d.year);
  // quadrants
  qg.select('rect')
    .transition().duration(600)
    .attr('fill', quad => d3.interpolateReds(colorScale(d[quad.key])));
  qg.select('text.value')
    .transition().duration(600)
    .text(quad => d[quad.key]);

  // move dot
  dot.transition().duration(800)
     .attr('cx', x(d.year));

  // highlight tick
  timeline.selectAll('text')
    .attr('fill', yy => yy === d.year ? '#000' : '#888');
}

function step(){
  update(idx);
  idx = (idx + 1) % data.length;
  setTimeout(step, 2000);
}

// kick it off
update(0);
setTimeout(step, 2000);
EOF

echo "✔️  setup complete!"
echo "To run your SWOT viz, simply:\n   cd $(pwd)\n   http-server .\nThen open http://localhost:8080 in your browser."
