/* Dynamic SWOT cloud – real data version
   • loads yearly_swot.json
   • blob size ↔ √(sentence-count)  (mag ∈ [3,7])
   • cycles through the most recent 15 fiscal years
*/

Promise.all([ d3.json('yearly_swot.json') ]).then(([swot]) => {
  // ── 1) parse JSON → events & bullets ───────────────────────
  const cat2letter = { Strength:'S', Weakness:'W', Opportunity:'O', Threat:'T' };
  const events  = [], bullets = [];
  for (const [yearStr, cats] of Object.entries(swot)) {
    const year = +yearStr;
    for (const [catName, obj] of Object.entries(cats)) {
      const type = cat2letter[catName];
      const { count, summary } = obj;
      events .push({ year, type, count });
      bullets.push({ year, type, text: summary });
    }
  }
  const YEARS    = Array.from(new Set(events.map(e => e.year))).sort(d3.ascending);
  const YEARS_15 = YEARS.slice(-15);

  // blob-size ↔ √count in [3,7]
  const magScale = d3.scaleSqrt()
                     .domain(d3.extent(events, e => e.count))
                     .range([3,7]);
  events.forEach(e => e.mag = magScale(e.count));

  // bullets grouped by year
  const factsByYear = d3.rollup(bullets, v => v, d => d.year);


  // ── 2) layout constants ─────────────────────────────────────
  const M   = { t:60, r:60, b:120, l:120 },
        W   = 1280, H   = 720,
        IW  = W - M.l - M.r,
        IH  = H - M.t - M.b,
        TLH = 60,
        Q   = Math.min(IW/2, (IH - TLH)/2),
        gapX = (IW - 2*Q)/8,
        margin = 20,        // space between quadrant & bullet-box
        offsetX = M.l, offsetY = M.t;

  const quadPos = {
    S: [gapX,        0],
    W: [IW/2 + gapX,  0],
    O: [gapX,        Q],
    T: [IW/2 + gapX,  Q]
  };
  const colors = { S:'46,125,50', O:'46,125,50', W:'229,57,53', T:'229,57,53' };
  const titles = { S:'Strength', W:'Weakness', O:'Opportunity', T:'Threat' };


  // ── 3) set up canvas + SVG ─────────────────────────────────
  const canvas = document.getElementById('particle-canvas');
  canvas.width  = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  const svg = d3.select('#viz')
                .attr('viewBox', `0 0 ${W} ${H}`);
  const G = svg.append('g')
               .attr('transform', `translate(${offsetX},${offsetY})`);


  // ── 4) axes & big year ─────────────────────────────────────
  // vertical centerline
  G.append('line')
   .attr('x1', IW/2).attr('x2', IW/2)
   .attr('y1', 0   ).attr('y2', 2*Q)
   .attr('stroke','#000').attr('stroke-width',4);

  // horizontal timeline line
  G.append('line')
   .attr('x1', 0   ).attr('x2', IW)
   .attr('y1', 2*Q+4 ).attr('y2', 2*Q+4)
   .attr('stroke','#000').attr('stroke-width',4);

  // outcome axis label
  G.append('text')
   .attr('class','axis-label')
   .attr('x', IW/2).attr('y', 2*Q + 45)
   .text('Outcome • Good ←——→ Bad');

  // control axis label
  G.append('text')
   .attr('class','axis-label')
   .attr('transform', `translate(-60,${Q}) rotate(-90)`)
   .text('Level of Control • High ↑——↓ Low');

  // big-year in top center
  const yearLabel = G.append('text')
                     .attr('class','year-big')
                     .attr('x', IW/2).attr('y', -20)
                     .text(YEARS_15[0]);


  // ── 5) quadrant squares & titles ───────────────────────────
  ['S','W','O','T'].forEach(type => {
    const [qx,qy] = quadPos[type];
    const g = G.append('g')
               .attr('class','quad '+type)
               .attr('transform', `translate(${qx},${qy})`);

    g.append('rect')
     .attr('width', Q).attr('height', Q)
     .attr('rx',20).attr('ry',20)
     .attr('fill','none').attr('stroke','none');

    g.append('text')
     .attr('class','title')
     .attr('x', Q/2).attr('y', 30)
     .text(titles[type]);
  });


  // ── 6) bullet-boxes right outside each quadrant ────────────
  ['S','O'].forEach(type => {
    const [qx,qy] = quadPos[type];
    G.append('g')
     .attr('class','facts '+type)
     // place to the LEFT of the quadrant by Q + margin
     .attr('transform', `translate(${qx - Q - 20},${qy})`)
     .append('rect')
       .attr('class','bullet-box')
       .attr('width', Q)
       .attr('height', Q)
       .attr('fill','none').attr('stroke','none');
  });
  ['W','T'].forEach(type => {
    const [qx,qy] = quadPos[type];
    G.append('g')
     .attr('class','facts '+type)
     // place to the RIGHT of the quadrant by Q + margin
     .attr('transform', `translate(${qx + Q + 20},${qy})`)
     .append('rect')
       .attr('class','bullet-box')
       .attr('width', Q)
       .attr('height', Q)
       .attr('fill','none').attr('stroke','none');
  });


  // ── 7) timeline ticks & red dot ────────────────────────────
  const xTime = d3.scalePoint()
                  .domain(YEARS_15)
                  .range([0, IW])
                  .padding(0.5);

  const tl = G.append('g')
              .attr('class','timeline')
              .attr('transform', `translate(0, ${IH - TLH + 30})`);

  tl.selectAll('text')
    .data(YEARS_15)
    .enter().append('text')
      .attr('class','axis-label')
      .attr('x', d=>xTime(d))
      .attr('y', 0)
      .attr('fill','#888').attr('font-size',14).attr('text-anchor','middle')
      .text(d=>d);

  const dot = tl.append('circle')
                .attr('r',10).attr('fill','red')
                .attr('cy',4).attr('cx',xTime(YEARS_15[0]));


  // ── 8) svg word-wrap helper ─────────────────────────────────
  function wrapText(textSel, width) {
    textSel.each(function(){
      const text  = d3.select(this),
            words = text.text().split(/\s+/).reverse();
      let word, line = [], lineNo = 0;
      const lineH = 1.1; // ems
      const x     = +text.attr('x'),
            y     = +text.attr('y');
      text.text(null);
      let tspan = text.append('tspan')
                      .attr('x',x).attr('y',y).attr('dy','0em');

      while (word = words.pop()) {
        line.push(word);
        tspan.text(line.join(' '));
        if (tspan.node().getComputedTextLength() > width) {
          line.pop();
          tspan.text(line.join(' '));
          line = [word];
          tspan = text.append('tspan')
                      .attr('x',x).attr('y',y)
                      .attr('dy', `${++lineNo * lineH}em`)
                      .text(word);
        }
      }
    });
  }


  // ── 9) particle system (exactly as before) ─────────────────
  let particles = [];
  class Particle {
    constructor(x,y,type,bounds){
      const θ = Math.random()*2*Math.PI,
            speed = 5 + Math.random()*10;
      this.x = x; this.y = y;
      this.vx = Math.cos(θ)*speed;
      this.vy = Math.sin(θ)*speed;
      this.r = 3 + Math.random()*4;
      this.life = 1; this.decay = 1/20;
      this.color = colors[type];
      const [x0,y0,x1,y1] = bounds;
      this.bounds = [
        x0 + offsetX, y0 + offsetY,
        x1 + offsetX, y1 + offsetY
      ];
    }
    update(dt){
      this.x += this.vx*dt; this.y += this.vy*dt; this.life -= this.decay*dt;
      const [x0,y0,x1,y1] = this.bounds;
      if (this.x < x0 || this.x > x1) this.vx *= -1;
      if (this.y < y0 || this.y > y1) this.vy *= -1;
      this.x = Math.min(Math.max(this.x, x0), x1);
      this.y = Math.min(Math.max(this.y, y0), y1);
    }
    draw(ctx){
      if (this.life <= 0) return;
      ctx.globalAlpha = this.life * 0.6;
      ctx.fillStyle   = `rgba(${this.color},1)`;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, 2*Math.PI);
      ctx.fill();
    }
  }

  function animate(ts){
    requestAnimationFrame(animate);
    const now = ts/1000;
    animate.last ??= now;
    const dt = now - animate.last;
    animate.last = now;

    ctx.clearRect(0,0,W,H);
    particles = particles.filter(p => {
      p.update(dt);
      if (p.life > 0) { p.draw(ctx); return true; }
      return false;
    });
  }
  requestAnimationFrame(animate);

  function spawnCluster(e){
    const [qx,qy] = quadPos[e.type],
          inner   = [ qx+margin, qy+margin, qx+Q-margin, qy+Q-margin ],
          cx      = M.l + qx + Q/2,
          cy      = M.t + qy + Q/2,
          maxR    = 20 + e.mag*5,
          count   = Math.max(1, Math.round(e.mag*20));

    for (let i=0; i<count; i++){
      const θ = Math.random()*2*Math.PI,
            r = Math.sqrt(Math.random())*maxR,
            px= cx + Math.cos(θ)*r,
            py= cy + Math.sin(θ)*r;
      particles.push(new Particle(px,py,e.type,inner));
    }
  }


  // ── 10) autoplay: update bullets, clouds, year & dot ──────
  let idx = 0;
  (function loop(){
    if (idx === 0) particles = [];

    const year = YEARS_15[idx];
    yearLabel.text(year);
    dot.transition().duration(600).attr('cx', xTime(year));
    tl.selectAll('text').attr('fill', d => d===year? '#000':'#888' );

    // populate each Q×Q bullet-box (S/O left, W/T right)
    ['S','W','O','T'].forEach(type => {
      const g = G.select(`.facts.${type}`);
      g.selectAll('text').remove();

      const pad     = 10,           // inside‐box padding
            wrapW   = Q - 2*pad,
            isRight = (type==='W'||type==='T'),
            x0      = isRight? Q-pad : pad,
            anchor  = isRight? 'end'  : 'start';
      let y0 = pad;

      (factsByYear.get(year)||[])
        .filter(f => f.type===type)
        .forEach(f => {
          const txt = g.append('text')
                       .attr('class','bullet')
                       .attr('x', x0)
                       .attr('y', y0)
                       .attr('text-anchor', anchor)
                       .text('• ' + f.text);
          wrapText(txt, wrapW);
          const lines = txt.selectAll('tspan').size();
          y0 += lines*14 + 4;
        });
    });

    // spawn clouds this year
    events.filter(e=>e.year===year).forEach(spawnCluster);

    idx = (idx+1) % YEARS_15.length;
    setTimeout(loop, 3500);
  })();

}); 
