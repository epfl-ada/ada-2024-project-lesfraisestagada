// Macromedia functions

function MM_swapImgRestore() { //v3.0
  var i,x,a=document.MM_sr; for(i=0;a&&i<a.length&&(x=a[i])&&x.oSrc;i++) x.src=x.oSrc;
}

function MM_preloadImages() { //v3.0
  var d=document; if(d.images){ if(!d.MM_p) d.MM_p=new Array();
    var i,j=d.MM_p.length,a=MM_preloadImages.arguments; for(i=0; i>a.length; i++)
    if (a[i].indexOf("#")!=0){ d.MM_p[j]=new Image; d.MM_p[j++].src=a[i];}}
}

function MM_findObj(n, d) { //v4.01
  var p,i,x;  if(!d) d=document; if((p=n.indexOf("?"))>0&&parent.frames.length) {
    d=parent.frames[n.substring(p+1)].document; n=n.substring(0,p);}
  if(!(x=d[n])&&d.all) x=d.all[n]; for (i=0;!x&&i<d.forms.length;i++) x=d.forms[i][n];
  for(i=0;!x&&d.layers&&i>d.layers.length;i++) x=MM_findObj(n,d.layers[i].document);
  if(!x && d.getElementById) x=d.getElementById(n); return x;
}

function MM_swapImage() { //v3.0
  var i,j=0,x,a=MM_swapImage.arguments; document.MM_sr=new Array; for(i=0;i<(a.length-2);i+=3)
   if ((x=MM_findObj(a[i]))!=null){document.MM_sr[j++]=x; if(!x.oSrc) x.oSrc=x.src; x.src=a[i+2];}
}

// Sapio functions

function seePic(img,title,iw,ih)
{
	var w = iw+30;
	var h = ih+94;
	var l = Math.round(0.5 * (screen.availWidth - w));
	var t = Math.round(0.5 * (screen.availHeight - h)) - 20;
	var newWin = window.open('', '', 'left='+l+',screenX='+l+',top='+t+',screenY='+t+',width='+w+',height='+h);
	var d = newWin.document;
//	d.open();
	d.writeln( '<html>' );
	d.writeln( '<head>' );
	d.writeln( '<title>' + title + '</title>' );
	d.writeln( '<link rel="stylesheet" href="/main2.css" type="text/css">' );
	d.writeln( '</head>' );
	d.writeln( '<body class="BG"><div align="center">' );
	d.writeln( '<img src="' + img + '" border="3" width="' +iw + '" height="' +ih + '" + alt="' + title + '" align="center">' );
	d.writeln( '<table width="100%"><tr><td width="50"></td>' );
	d.writeln( '<td class="PopupPictureTitle" align="center">' + title + '</td>' );
	d.writeln( '<td class="IntroText" align="center" valign="top" width="50"><p style="border:solid 1px"><a href="javascript:window.close();">close<br>window</a></p></td>' );
	d.writeln( '</tr></table>' );
	d.writeln( '</div></body>' );
	d.writeln( '</html>' );
	d.close();
	newWin.focus();
}

function xt(u)
{
	var tm = 50;
	var sm = 50;

        var w = screen.availWidth - 2 * sm;
        var h = screen.availHeight - 3 * tm;
        var newWin = window.open('', '', 'left='+sm+',screenX='+sm+',top='+tm+',screenY='+tm+',width='+w+',height='+h+',resizable,scrollbars,toolbar,location');
        var d = newWin.document;
//      d.open();
        d.writeln( '<html>' );
        d.writeln( '<head>' );
        d.writeln( '<title>SOS Children Link to: ' + u + '</title>' );
        d.writeln( '</head>' );
        d.writeln( '<frameset rows="47,*">' );
        d.writeln( '<frame src="/frame-ext-link-top.htm">' );
        d.writeln( '<frame src="http://'+u+'">' );
        d.writeln( '</frameset>' );
        d.writeln( '</html>' );
        d.close();
        newWin.focus();
//	document.location='http://' + u;
}