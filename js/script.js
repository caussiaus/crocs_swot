Promise.all([
  d3.json('events.json'),
  d3.json('bulletpoints.json')
]).then(([events, bullets]) => {

  // ── Layout & scales ───────────────────────────────
  const YEARS = Array.from(new Set(events.map(e=>e.year))).sort(d3.ascending);
  const M   = { t:60, r:60, b:120, l:120 },
        W   = 1280, H = 720,
        IW  = W - M.l - M.r,
        IH  = H - M.t - M.b,
        TL_H= 60;
  const Q    = Math.min(IW/2, (IH - TL_H)/2),
        gapX = (IW - 2*Q)/3,
        margin = 20;
  const offsetX = M.l, offsetY = M.t;
  const quadPos = {
    S: [gapX,     0],
    W: [gapX*2+Q, 0],
    O: [gapX,     Q],
    T: [gapX*2+Q, Q]
  };
  const colors = { S:'46,125,50', O:'46,125,50', W:'229,57,53', T:'229,57,53' };
  const factsByYear = new Map(bullets.map(d=>[d.year,d.facts]));
  const xTime = d3.scalePoint().domain(YEARS).range([0,IW]).padding(0.5);

  // ── Canvas setup ───────────────────────────────────
  const canvas = document.getElementById('particle-canvas');
  canvas.width = W;  canvas.height = H;
  const ctx = canvas.getContext('2d');

  // ── SVG setup ───────────────────────────────────────
  const svg = d3.select('#viz');
  const G = svg.append('g')
    .attr('transform', `translate(${offsetX},${offsetY})`);

  // draw centerline
  G.append('line')
    .attr('x1',IW/2).attr('x2',IW/2)
    .attr('y1',0).attr('y2',2*Q)
    .attr('stroke','#000').attr('stroke-width',4);

  // quadrants
  const titles = { S:'Strength', W:'Weakness', O:'Opportunity', T:'Threat' };
  ['S','W','O','T'].forEach(type => {
    const [qx,qy] = quadPos[type];
    const group = G.append('g')
      .attr('class','quad '+type)
      .attr('transform', `translate(${qx},${qy})`);

    group.append('rect')
      .attr('width', Q).attr('height', Q)
      .attr('rx',20).attr('ry',20)
      .attr('fill','none')         // transparent!
      .attr('stroke','#ccc')
      .attr('stroke-width',2);

    group.append('text')
      .attr('class','title')
      .attr('x', Q/2).attr('y', 30)
      .text(titles[type]);

    const fx = (type==='S'||type==='O') ? qx-140 : qx+Q+20;
    const fy = qy + Q/2 - 20;
    G.append('g')
      .attr('class','facts '+type)
      .attr('transform', `translate(${fx},${fy})`);
  });

  // axes & timeline
  G.append('line')
    .attr('x1',0).attr('x2',IW)
    .attr('y1',2*Q+4).attr('y2',2*Q+4)
    .attr('stroke','#000').attr('stroke-width',4);

  G.append('text')
    .attr('class','axis-label')
    .attr('x',IW/2).attr('y',2*Q+45)
    .text('Outcome • Good ←——→ Bad');

  G.append('text')
    .attr('class','axis-label')
    .attr('transform',`translate(-60,${Q}) rotate(-90)`)
    .text('Level of Control • High ↑——↓ Low');

  const yearLabel = G.append('text')
    .attr('class','year-big')
    .attr('x',IW/2).attr('y',-20)
    .text(YEARS[0]);

  const tl = G.append('g')
      .attr('class','timeline')
      .attr('transform',`translate(0,${IH-TL_H+30})`);

  tl.selectAll('text')
    .data(YEARS)
    .enter().append('text')
      .attr('x',d=>xTime(d))
      .attr('y',0)
      .attr('fill','#888')
      .attr('font-size',14)
      .attr('text-anchor','middle')
      .text(d=>d);

  const dot = tl.append('circle')
      .attr('r',10).attr('fill','red')
      .attr('cy',0).attr('cx',xTime(YEARS[0]));

  // ── Particle system ────────────────────────────────
  let particles = [];
  class Particle {
    constructor(x,y,type,bounds){
      this.x = x; this.y = y; this.type = type;
      const a = Math.random()*2*Math.PI,
            speed = 5 + Math.random()*10;
      this.vx = Math.cos(a)*speed;
      this.vy = Math.sin(a)*speed;
      this.r  = 3 + Math.random()*4;
      this.life  = 1;
      this.decay = 1/20;
      this.color = colors[type];
      const [x0,y0,x1,y1] = bounds;
      this.bounds = [
        x0 + offsetX, y0 + offsetY,
        x1 + offsetX, y1 + offsetY
      ];
    }
    update(dt){
      this.x += this.vx*dt; this.y += this.vy*dt;
      this.life -= this.decay*dt;
      const [x0,y0,x1,y1] = this.bounds;
      if (this.x < x0){ this.x=x0; this.vx*=-1; }
      if (this.x > x1){ this.x=x1; this.vx*=-1; }
      if (this.y < y0){ this.y=y0; this.vy*=-1; }
      if (this.y > y1){ this.y=y1; this.vy*=-1; }
    }
    draw(ctx){
      if (this.life <= 0) return;
      ctx.globalAlpha = this.life * 0.6;
      ctx.fillStyle = `rgba(${this.color},1)`;
      ctx.beginPath();
      ctx.arc(this.x,this.y,this.r,0,2*Math.PI);
      ctx.fill();
    }
  }

  function animate(ts){
    requestAnimationFrame(animate);
    const now = ts/1000;
    animate.last = animate.last||now;
    const dt = now - animate.last;
    animate.last = now;

    ctx.clearRect(0,0,W,H);
    particles = particles.filter(p => {
      p.update(dt);
      if (p.life>0){ p.draw(ctx); return true; }
      return false;
    });
  }
  requestAnimationFrame(animate);

  function spawnCluster(e){
    const [qx,qy] = quadPos[e.type];
    const boundsInner = [
      qx + margin,      qy + margin,
      qx + Q - margin,  qy + Q - margin
    ];
    const cx = M.l + qx + Q/2;
    const cy = M.t + qy + Q/2;
    const maxR = 20 + e.mag*5;
    const count= Math.max(1, Math.round(e.mag*20));
    for(let i=0;i<count;i++){
      const θ = Math.random()*2*Math.PI;
      const r = Math.sqrt(Math.random())*maxR;
      const px= cx + Math.cos(θ)*r,
            py= cy + Math.sin(θ)*r;
      particles.push(new Particle(px,py,e.type,boundsInner));
    }
  }

  // ── Render & autoplay ───────────────────────────────
  let idx = 0;
  (function loop(){
    // on wrap-around, clear existing clouds
    if(idx === 0) particles = [];
    // render this year
    const year = YEARS[idx];
    // bullets
    const facts = factsByYear.get(year) || [];
    ['S','W','O','T'].forEach(t => {
      const g = G.select(`.facts.${t}`);
      g.selectAll('text').remove();
      facts.filter(f=>f.type===t).forEach((f,i)=>{
        g.append('text')
          .attr('class','bullet')
          .attr('x',0).attr('y',i*18)
          .text('• '+f.text);
      });
    });
    // spawn new puff events
    events.filter(e=>e.year===year).forEach(spawnCluster);

    // update labels
    yearLabel.text(year);
    dot.transition().duration(600).attr('cx',xTime(year));
    tl.selectAll('text').attr('fill', d=>d===year?'#000':'#888');

    // advance
    idx = (idx + 1) % YEARS.length;
    setTimeout(loop, 3500);
  })();

});
